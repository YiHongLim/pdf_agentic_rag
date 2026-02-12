from contextlib import asynccontextmanager
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from openai import OpenAI

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()


index = None


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

    # 3) Build vector index
    # This:
    #   - Splits documents into chunks
    #   - Creates embeddings (vectors) for each chunk
    #   - Stores them in memory for fast retrieval
    index = VectorStoreIndex.from_documents(documents)

    yield


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
