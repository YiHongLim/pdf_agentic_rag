from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter

from typing import Dict, Any, List, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


index = None


# Agent state (tracks conversation)
class AgentState(TypedDict):
    messages: List[Any]
    retrieved_chunks: List[Dict]
    needs_retrieval: bool


def extract_term(question: str) -> str:
    """Very simple heuristic: use the question text without '?' as the term."""
    q = question.strip()
    if q.endswith("?"):
        q = q[:-1]
    return q.strip()


# Your existing RAG as a LangGraph TOOL
@tool
def rag_retrieve(question: str) -> str:
    """Retrieve relevant chunks from the RAG index."""
    if index is None:
        return "RAG is not ready."

    query_engine = index.as_query_engine(similarity_top_k=15)
    response = query_engine.query(question)

    chunks = []
    for node in response.source_nodes:
        file_name = node.metadata.get("file_name", "Unknown")
        if "/" in file_name:
            file_name = file_name.split("/")[-1]
        chunks.append(
            {
                "text": node.text[:300] + "...",
                "file": file_name,
                "score": node.score or 0,
            }
        )

    term = extract_term(question)
    term_lower = term.lower()

    keyword_hits = []
    others = []

    for c in chunks:
        if term_lower and term_lower in c["text"].lower():
            keyword_hits.append(c)
        else:
            others.append(c)

    ordered = keyword_hits + others
    top_chunks = ordered[:10]

    return top_chunks


# Router agent (decides if retrieval is needed)
router_llm = ChatOpenAI(model="gpt-4o-mini")
router_prompt = ChatPromptTemplate.from_template("""
You are a router for an agentic RAG system.
                                                 
Given a question, decide if it needs document retrieval:

RETRIEVE if:
- The question asks about a term, concept, or acronym that you do NOT confidently recognize from general knowledge, OR
- It needs specific facts, data, research findings, or content that could be inside the documents.

DIRECT if:
- You are confident you can answer from general knowledge alone,
- AND the question does not need any specific information from the documents.
                                    
Question: {question}
                                                 
Respond ONLY with: "RETRIEVE" or "DIRECT"
""")

router_chain = router_prompt | router_llm

# Answer agent
answer_llm = ChatOpenAI(model="gpt-4o-mini")
answer_prompt = ChatPromptTemplate.from_template("""
Answer the question using the provided context if available.
                                                 
Question: {question}
Context: {context}
                                                 
Answer:""")

answer_chain = answer_prompt | answer_llm


def create_agent_graph() -> StateGraph:
    """Create the agent RAG graph."""
    graph = StateGraph(AgentState)

    # Router node
    def router(state: AgentState) -> AgentState:
        question = state["messages"][-1].content
        decision = router_chain.invoke({"question": question})
        needs_retrieval = "RETRIEVE" in decision.content

        return {
            "messages": state["messages"]
            + [
                AIMessage(
                    content=f"Router: {'RETRIEVE' if needs_retrieval else 'DIRECT'}"
                )
            ],
            "needs_retrieval": needs_retrieval,
            "retrieved_chunks": [],
        }

    # Retrieval node
    def retrieve(state: AgentState) -> AgentState:
        question = state["messages"][-1].content
        chunks = rag_retrieve.invoke(question)

        preview = (
            "\n".join(
                f"{c['file']} (score: {round(c['score'], 3)}): {c['text'][:150]}..."
                for c in chunks
            )
            or "No chunks retrieved."
        )

        return {
            **state,
            "messages": state["messages"]
            + [
                AIMessage(
                    content=f"RAG Tool retrieved {len(chunks)} chunks:\n{preview}"
                )
            ],
            "retrieved_chunks": chunks,
        }

    def answer(state: AgentState) -> AgentState:
        question = [m for m in state["messages"] if isinstance(m, HumanMessage)][
            -1
        ].content

        retrieved = state.get("retrieved_chunks", [])

        if retrieved:
            top_texts = [c["text"] for c in retrieved[:3]]
            context = "\n\n".join(top_texts)

            if "page" in retrieved[0]:
                source_info = ", ".join(
                    {f"{c['file']} (p. {c['page']})" for c in retrieved[:3]}
                )

            else:
                source_info = ", ".join({c["file"] for c in retrieved[:3]})
            source_suffix = f"\n\nSources: {source_info}"
        else:
            context = "No documents were retrieved. Answer using general knowledge."
            source_suffix = ""

        # if state.get("retrieved_chunks"):
        #     top_texts = [c["text"] for c in state["retrieved_chunks"][:3]]
        #     context = "\n\n".join(top_texts)
        # else:
        #     context = "No documents were retrieved. Answer using general knowledge."

        answer = answer_chain.invoke({"question": question, "context": context})

        final_text = f"Final Answer: {answer.content}{source_suffix}"

        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=final_text)],
        }

    graph.add_node("router", router)
    graph.add_node("retrieve", retrieve)
    graph.add_node("answer", answer)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        lambda state: "retrieve" if state["needs_retrieval"] else "answer",
        {"retrieve": "retrieve", "answer": "answer"},
    )
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


agent_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager - handles startup and shutdown logic.

    Everything BEFORE yield = runs ONCE when server starts
    Everything AFTER yield = runs ONCE when server shuts down
    """
    global index

    # 1) Configure LlamaIndex
    # This tells LlamaIndex which LLM and embedding model to use
    Settings.llm = OpenAI(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",  # Converts text to vectors
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # 2) Load documents with metadata
    # SimpleDirectoryReader reads all files in a directory
    documents = SimpleDirectoryReader(
        "../pdfs", file_metadata=lambda filename: {"file_name": filename}
    ).load_data()
    print(f"Loaded {len(documents)} documents into RAG index")

    node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    # 3) Build vector index
    # This:
    #   - Splits documents into chunks
    #   - Creates embeddings (vectors) for each chunk
    #   - Stores them in memory for fast retrieval
    index = VectorStoreIndex.from_documents(documents, transformations=[node_parser])

    global agent_graph
    agent_graph = create_agent_graph()

    print("Agentic RAG graph initialized")
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """
    Health check endpoint - test if server is running
    Returns; JSON with status and whether index is loaded
    """
    return {"status": "ok", "index_loaded": index is not None}


@app.post("/query")
async def query(question: str):
    """
    RAG query endpoint - core of system

    Args:
        question: User's question as a query parameter
                Example: /query?question=What is RAG?

    Returns:
        JSON with answer and source excerpts
    """
    if index is None:
        return {"error": "Index not ready yet"}

    # Step 1: Create query engine with more retrieved chunk (default is 2)
    # Handles retrieval -> ranking -> generation
    query_engine = index.as_query_engine(
        similarity_top_k=3  # Retrieve top 3 most relevant chunks
    )

    # Step 2: Query the index
    # Behind the scenes:
    #   1. Embeds your question
    #   2. Finds most similar chunks in vector DB
    #   3. Sends chunks + question to LLM
    #   4. LLM generates answer based on retrieved context
    response = query_engine.query(question)

    # Step 3: Extract enhanced sources with metadata
    # response.source_nodes = list of chunks used to generate answer
    sources = []
    for i, node in enumerate(response.source_nodes, 1):
        file_name = node.metadata.get("file_name", "Unknown")
        if "/" in file_name:
            file_name = file_name.split("/")[-1]  # Get filename, not full path
        sources.append(
            {
                "rank": i,
                "text": node.text[:300] + "...",
                "score": round(node.score, 4) if node.score else None,
                "file_name": file_name,
                "page": node.metadata.get("page_label", "N/A"),
            }
        )

    return {
        "answer": str(response),
        "question": question,
        "num_sources": len(sources),
        "sources": sources,
    }


@app.post("/agent-query")
async def agent_query(question: str):
    """Agentic RAG with LangGraph."""
    if agent_graph is None:
        return {"error": "Agent not ready"}

    result = agent_graph.invoke({"messages": [HumanMessage(content=question)]})

    final_messages = result["messages"]
    final_answer = final_messages[-1].content

    return {
        "question": question,
        "agent_steps": [m.content for m in final_messages],
        "final_answer": final_answer,
    }


@app.post("/test-chunks")
async def test_chunk_sizes(question: str):
    """
    Test different chunk sizes on the same question.
    Returns comparison of retrieval quality across sizes.
    """
    if index is None:
        return {"error": "Index not ready"}

    chunk_configs = [
        {"size": 256, "overlap": 32, "label": "Small"},
        {"size": 512, "overlap": 64, "label": "Medium"},
        {"size": 1024, "overlap": 128, "label": "Large"},
    ]

    results = {}

    for config in chunk_configs:
        print(f"🔄 Testing {config['label']} chunks ({config['size']} tokens)...")
        node_parser = SentenceSplitter(
            chunk_size=config["size"], chunk_overlap=config["overlap"]
        )

        temp_documents = SimpleDirectoryReader("../pdfs").load_data()
        temp_index = VectorStoreIndex.from_documents(
            temp_documents, transformations=[node_parser]
        )

        query_engine = temp_index.as_query_engine(similarity_top_k=2)
        response = query_engine.query(question)

        scores = [node.score for node in response.source_nodes if node.score]
        avg_score = sum(scores) / len(scores) if scores else 0

        estimated_chunks = sum(
            len(doc.text) // config["size"] + 1 for doc in temp_documents
        )

        results[config["label"]] = {
            "chunk_size": config["size"],
            "avg_similarity": round(avg_score, 4),
            "estimated_chunks": estimated_chunks,
            "retrieved_files": list(
                set(
                    node.metadata.get("file_name", "") for node in response.source_nodes
                )
            ),
        }

    best_result = max(results.items(), key=lambda x: x[1]["avg_similarity"])

    return {
        "question": question,
        "chunk_size_comparison": results,
        "best_config": best_result[1],
        "recommendation": f"Use {best_result[0]} chunks ({best_result[1]['chunk_size']} tokens)",
    }
