## Functional Requirements

1. Document Ingestion
  - The system shall load all PDF files from a configured directory (`../pdfs`).
  - The system shall extract text from each PDF and split it into smaller chunks suitable for retrieval.
2. Indexing
  - The system shall generate vector embeddings for each text chunk using an embedding model (OpenAI text-embedding-3-small).
  - The system shall build and store a vector index over all document chunks at application startup.
3. Question Answering (RAG)
  - The system shall expose a REST endpoint `POST /query` that accepts a natural-language question as input.
  - For each query, the system shall retrieve the top *k* most relevant document chunks from the index.
  - The system shall generate an answer using an LLM (GPT-4o-mini) conditioned on the retrieved chunks.
  - The system shall return both the generated answer and metadate for retrieved chunks (filename, page, similarity score).
4. Health Monitoring
  - The system shall expose a `GET /health` endpoint that reports whether the RAG index has been successfully loaded.
5. Source Transparency
  - The system shall return at least one relevant source chunk for every answer, allowing users to verify where the answer came from.

6. Error handling
  - If the index is not yet initialized, the system shall return a clear error message instead of attempting to answer.
  - If the LLM or embedding API call fails, the system shall return an error payload with a human-readable message.
  
## Technical Design

### Architecture Overview

- **Cient**: (for now) FASTAPI Swagger UI
- **API Layer**: FastAPI application exposing `/health` and `/query` endpoints.
- **RAG Engine**: LlamaIndex `VectorStoreIndex` providing retrieval + generation.
- **LLM & Embeddings**: OpenAI GPT-4o-mini for generation, `text-embedding-3-small` for embeddings.
- **Storage**: In-memory vector index (backed by LlamaIndex / Chroma-style vector store).
- **Configguration / Secrets**: Environment variables loaded from `.env` via `python-dotenv`

### Lifecycle Management
- The FastAPI app uses a **lifespan context manager**:
  - On startup:
    - Load environment variables (including `OPENAI_API_KEY`).
    - Configure `Settings.llm)` (GPT-4o-mini) and `Settings.embed_model` (text-embedding-3-small).
    - Use `SimpleDirectoryReader("../pdfs", file_metadata=...)` to load and parse all PDFs into document chunks with metadata (filename, page label).
    - Build a `VectorStoreIndex` from the documents and store it in a global `index` variable.
    - On shutdown:
      - Log shutdown and clean up resources if needed
      
### Data Flow

1. **Indexing Phase (startup)**
   - Read PDFs from `./pdfs`.
   - Extract text and split into chunks.
   - Generate embeddings for each chunk via OpenAI embeddings API.
   - Store embeddings and metadata in the vector index.
2. **Query Phase (runtime)**
   - Client sends `POST /query?question=...`.
   - Backend creates a query engine via `index.as_query_engine(similarity_top_k=3)`.
   - Query engine:
     - Embeds the question.
     - Performs similarity search over document vectors.
     - Retrieves the top-k chunks with scores.
     - Calls GPT-4o-mini with the question and retrieved context.
   - Backend returns:
     - `answer`: generated text.
     - `sources`: list of objects `{ rank, text, score, file_name, page }`.

### Key Endpoints
- `GET /health`
  - Purpose: Simple readiness probe.
  - Response: `{ "status": "ok", "index_loaded": true | false }`.
- `POST /query`
  - Input: query parameter `question: str`.
  - Processing:
    - Validate that index is initialized.
    - Run retrieval + generation via LlamaIndex query engine.
    - Collect `response.source_nodes` and extract text, similarity scores, and metadata.
    - Output:
      ```json
      {
        "answer": "<string>",
        "question": "<original question>",
        "num_sources": 3,
        "sources": [
          {
            "rank": 1,
            "text": "...",
            "score": 0.72,
            "file_name": "2601.02749v1.pdf",
            "page": "7"
          }
        ]
      }
      ```

# Technology Stack 
- **Language**: Python 3.12
- **Backend Framework**: FastAPI + Uvicorn
- **RAG Framework**: LlamaIndex (core, llms-openai, embeddings-openai)
- **LLM & Embeddings**: OpenAI GPT-4o-mini, `text-embedding-3-small`
- **Env Management**: `python-dotenv`
- **Dependency Management**: `requirements.txt`
- **Version Control**: Git + GitHub
  
