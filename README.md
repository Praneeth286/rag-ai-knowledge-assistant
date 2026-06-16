# 🤖 RAG-based AI Knowledge Assistant

A production-ready Q&A system that answers questions over your own documents using
**retrieval-augmented generation (RAG)**. Built with **FastAPI**, **LangChain**,
**OpenAI GPT-4**, and a **FAISS** vector store, with streaming responses, conversation
memory, and source citations for transparent, trustable answers.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat&logo=meta&logoColor=white)

---

## ✨ Features

- **Document ingestion** with semantic chunking for higher retrieval relevance.
- **FAISS vector store** for fast, local similarity search over embeddings.
- **RAG query pipeline** that grounds GPT-4 answers in retrieved context.
- **Source citations** returned with every answer to reduce hallucination.
- **Streaming responses** over Server-Sent Events for a responsive UX.
- **Conversation memory** so follow-up questions keep context.
- **Clean FastAPI service** ready to containerize and deploy (AWS Lambda / ECS).

---

## 🏗️ Architecture

```
                ┌──────────────┐     embeddings      ┌──────────────┐
  documents ──▶ │  Ingestion   │ ──────────────────▶ │ FAISS Vector │
                │  + chunking  │                     │    Store     │
                └──────────────┘                     └──────┬───────┘
                                                            │ top-k retrieval
   user query ──▶ FastAPI ──▶ RAG pipeline ──▶ GPT-4 ──▶ answer + citations
                                  ▲                           │
                                  └────── conversation memory ┘
```

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure your OpenAI key
cp .env.example .env
# edit .env and set OPENAI_API_KEY

# 3. Ingest documents (drop .txt/.md files into ./data first)
python rag.py --ingest ./data

# 4. Run the API
uvicorn main:app --reload
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 📡 API

| Method | Endpoint   | Description                                  |
|--------|------------|----------------------------------------------|
| `POST` | `/ingest`  | Index a batch of documents into FAISS        |
| `POST` | `/ask`     | Ask a question, returns answer + sources     |
| `POST` | `/ask/stream` | Same as `/ask` but streams tokens (SSE)   |
| `GET`  | `/health`  | Liveness probe                               |

Example:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our refund policy?"}'
```

---

## 🧱 Project structure

```
.
├── main.py            # FastAPI app & routes
├── rag.py             # Ingestion, chunking, embeddings, retrieval, generation
├── config.py          # Settings loaded from environment
├── requirements.txt
├── .env.example
└── data/              # Your source documents
```

---

## 🗺️ Roadmap

- [ ] Swap FAISS for a managed vector DB (pgvector / Pinecone) for scale
- [ ] Add an evaluation harness (answer relevance + faithfulness scoring)
- [ ] Re-ranking layer before generation
- [ ] Auth + per-tenant document isolation

---

## 📄 License

MIT
