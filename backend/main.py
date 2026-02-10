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

    # 2) Load documents
    # SimpleDirectoryReader reads all files in a directory
    documents = SimpleDirectoryReader("./pdfs").load_data()
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

    # Step 1: Create query engine from index
    # Handles retrieval -> ranking -> generation
    query_engine = index.as_query_engine()

    # Step 2: Query the index
    # Behind the scenes:
    #   1. Embeds your question
    #   2. Finds most similar chunks in vector DB
    #   3. Sends chunks + question to LLM
    #   4. LLM generates answer based on retrieved context
    response = query_engine.query(question)

    # Step 3: Extract source chunks (for citations)
    # response.source_nodes = list of chunks used to generate answer
    sources = []
    for node in response.source_nodes:
        sources.append(node.text[:300])

    return {
        "answer": str(response),
        "sources": sources,
    }
