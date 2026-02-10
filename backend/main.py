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
