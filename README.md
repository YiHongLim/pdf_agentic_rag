# PDF Agentic RAG 

A FastAPI service that answers questions over a local corpus of PDFs using:

- **LlamaIndex** for ingestion, chunking, and vector search
- **LangGraph + LangChain** for a stateful, agentic RAG workflow
- **OpenAI** models for LLMs and embeddings

The service exposes both a simple RAG endpoint and a richer agentic RAG endpoint with routing, grading, query rewriting, and document-grounded answers.

---

## Features

- 📄 PDF ingestion from a local directory (`../pdfs`)
- 🔍 Classic RAG endpoint (`/query`) with source citations
- 🤖 Agentic RAG endpoint (`/agent-query`) with:
  - LLM-based query router (DIRECT vs RETRIEVE)
  - Custom retrieval tool over LlamaIndex
  - LLM-based document grading (filters irrelevant chunks)
  - Optional query rewriting + retry loop
  - Answers that **either use docs or say “I don’t know from these docs”**
- 🧪 Chunk-size testing endpoint (`/test-chunks`) to compare retrieval quality across chunk sizes

---

## Architecture

### High-Level

- **FastAPI app** with a lifespan context:
  - Initializes LlamaIndex and LangGraph once at startup
  - Serves HTTP endpoints for health, simple RAG, agentic RAG, and chunk-size tests

- **LlamaIndex layer**:
  - Loads all PDFs from `../pdfs`
  - Adds `file_name` as metadata
  - Splits documents with `SentenceSplitter(chunk_size=512, chunk_overlap=64)`
  - Builds an in-memory `VectorStoreIndex` using OpenAI embeddings (`text-embedding-3-small`)

- **Agentic RAG layer (LangGraph)**:
  - State graph over an `AgentState` that tracks:
    - `messages`: conversation history (HumanMessage / AIMessage)
    - `retrieved_chunks`: list of chunk dicts (`text`, `file`, `score`, optional `page`)
    - `needs_retrieval`: router decision flag
    - `retry_count`: number of query-rewrite retries
  - Nodes:
    - `router` → decide DIRECT vs RETRIEVE
    - `retrieve` → call RAG tool over LlamaIndex
    - `grade` → LLM-based relevance filtering of chunks
    - `rewrite_query` → rewrite vague/failed questions (optional, with retry cap)
    - `answer` → generate final answer + sources, or say “I don’t know from these docs”

---

## Agent Workflow

### 1. Router

- Uses a small OpenAI chat model (`gpt-4o-mini`) with a routing prompt.
- For each question, outputs `"RETRIEVE"` or `"DIRECT"`.
- Sets `needs_retrieval` in `AgentState` and logs a message like `Router: RETRIEVE`.

```python
graph.add_conditional_edges(
    "router",
    lambda state: "retrieve" if state["needs_retrieval"] else "answer",
    {"retrieve": "retrieve", "answer": "answer"},
)
```

### 2. RAG Tool (`rag_retrieve`)

- Wraps the LlamaIndex `VectorStoreIndex` as a LangChain tool.
- Steps:
  - Run `index.as_query_engine(similarity_top_k=15).query(question)`
  - Build chunk dicts:
    - `text` = first 300 chars of node text
    - `file` = basename of `file_name` metadata
    - `score` = similarity score
  - Simple keyword boosting:
    - Extract `term` from question (strip `?` and whitespace)
    - Rank chunks where `term` appears in text first
  - Return top 10 ordered chunks

### 3. Retrieve Node

- Gets the last `HumanMessage` as `question`.
- Calls `rag_retrieve(question)`.
- Logs a preview of retrieved chunks as an `AIMessage`:
  - `RAG Tool retrieved N chunks:\n{file} (score: ...): {snippet}...`
- Stores chunks in `retrieved_chunks`.

### 4. Grade Node

- Uses a deterministic LLM (`gpt-4o-mini`, `temperature=0`) as a judge.
- For each chunk:
  - Asks: “Does this chunk help answer the question?”
  - Requires:
    - First line: `"yes"` or `"no"`
    - Second line: brief reason
  - Keeps only `"yes"` chunks.
- Logs:
  - `Grader: kept X of Y chunks.\n{per-chunk reasoning...}`
- Updates `retrieved_chunks` to the filtered list.

### 5. (Optional) Query Rewrite Node

- Activated when:
  - `retrieved_chunks` is empty after grading
  - `retry_count < MAX_RETRIES`
- Uses an LLM to rewrite the question into a concise search query.
- Logs: `Rewritten query: {rewritten}`.
- Increments `retry_count`, clears `retrieved_chunks`, and routes back to `retrieve`.

```python
graph.add_conditional_edges(
    "grade",
    route_after_grading,
    {"rewrite_query": "rewrite_query", "answer": "answer"},
)

graph.add_edge("rewrite_query", "retrieve")
```

### 6. Answer Node

- Gets last `HumanMessage` as `question`.
- If `retrieved_chunks` is non-empty:
  - Builds `context` from the top 3 chunk texts.
  - Aggregates source info from `file` (and `page` if present).
  - Calls `answer_chain` with `{question, context}`.
  - Returns `Final Answer: {answer}{Sources: ...}`.
- If `retrieved_chunks` is empty:
  - Does **not** answer from general knowledge.
  - Returns:
    - `Final Answer: I don't know from these docs. The retrieved documents do not contain enough information to answer this question reliably.`

---

## Endpoints

### `GET /health`

- Simple health check.
- Response:
  ```json
  {
    "status": "ok",
    "index_loaded": true
  }
  ```

### `POST /query`

- Simple RAG over LlamaIndex (no agent).
- Input:
  - `question: str` (as query parameter or body, depending on how you integrate).
- Behavior:
  - `similarity_top_k=3`
  - Returns answer + structured sources.
- Example response:
  ```json
  {
    "answer": "string",
    "question": "What is RAG?",
    "num_sources": 3,
    "sources": [
      {
        "rank": 1,
        "text": "chunk text...",
        "score": 0.89,
        "file_name": "paper.pdf",
        "page": "3"
      }
    ]
  }
  ```

### `POST /agent-query`

- Full agentic RAG with LangGraph.
- Input:
  - `question: str`
- Behavior:
  - Runs the LangGraph state machine:
    - `router → (retrieve → grade → rewrite? → retrieve → grade) → answer`
- Example response:
  ```json
  {
    "question": "what is universalRAG?",
    "agent_steps": [
      "what is universalRAG?",
      "Router: RETRIEVE",
      "RAG Tool retrieved 10 chunks: ...",
      "Grader: kept 3 of 10 chunks.\n...",
      "Rewritten query: ...",  // only if rewrite is triggered
      "Final Answer: ...\n\nSources: 2504.20734v3.pdf"
    ],
    "final_answer": "Final Answer: ...\n\nSources: 2504.20734v3.pdf"
  }
  ```

### `POST /test-chunks`

- Experiments with different chunk sizes for retrieval:
  - Small: 256 tokens
  - Medium: 512 tokens
  - Large: 1024 tokens
- For each config:
  - Rebuilds a temporary index
  - Queries with `similarity_top_k=2`
  - Computes average similarity score, estimated chunk count, and retrieved files
- Returns:
  ```json
  {
    "question": "...",
    "chunk_size_comparison": {
      "Small": {...},
      "Medium": {...},
      "Large": {...}
    },
    "best_config": {...},
    "recommendation": "Use Medium chunks (512 tokens)"
  }
  ```

---

## Setup & Running

1. **Install dependencies**

   ```bash
   pip install fastapi uvicorn llama-index langchain-openai langgraph python-dotenv
   ```

2. **Set environment variables**

   - Create a `.env` file with:
     ```env
     OPENAI_API_KEY=your_openai_api_key_here
     ```

3. **Prepare PDFs**

   - Put your PDF files under `../pdfs` relative to the app file.
   - Ensure the process has read access.

4. **Run the server**

   ```bash
   uvicorn main:app --reload
   ```

5. **Test**

   - Health: `GET /health`
   - Simple RAG: `POST /query?question=What is RAG?`
   - Agentic RAG: `POST /agent-query?question=what is universalRAG?`
   - Chunk test: `POST /test-chunks?question=...`

---

## Future Improvements

- Add **answer validation / reflection** node to check if answers are fully supported by the retrieved chunks.
- Introduce more tools (e.g., web search, SQL/vector DBs) and extend the router to choose between 
