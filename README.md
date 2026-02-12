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
  
