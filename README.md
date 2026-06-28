# 🔗 LangChain & RAG — Learning Repository

A hands-on exploration of two core concepts in modern AI application development:

1. **LangChain** — Building conversational AI chains with persistent cloud memory  
2. **RAG (Retrieval-Augmented Generation)** — Grounding LLM responses with real knowledge from a vector database

Both modules are production-oriented, using cloud services (no local-only dependencies) and exposed via **FastAPI** endpoints.

---

## 📁 Repository Structure

```
├── langchain_chat/            # Module 1: LangChain + Memory
│   ├── llm_chain.py           # Core LCEL chain (Groq LLM + prompt)
│   └── main.py                # FastAPI server with Upstash Redis memory
│
├── rag_pipeline/              # Module 2: RAG Pipeline
│   ├── knowledge.txt          # Knowledge base (astrophysics dataset)
│   ├── rag_pipeline.py        # Document ingestion into Pinecone
│   ├── rag_llm.py             # CloudRAGPipeline class (retrieve + generate)
│   └── rag_fast_api.py        # FastAPI server exposing RAG endpoint
│
├── .env.example               # Template for required environment variables
├── .gitignore
└── README.md
```

---

## 🧠 Module 1: LangChain Chat (`langchain_chat/`)

A conversational AI chatbot built with **LangChain Expression Language (LCEL)** that remembers past conversations using cloud-based memory.

### Key Concepts Covered
- **LCEL Chains** — Composing prompt → model → output parser using the `|` pipe syntax
- **Chat Memory** — Persistent conversation history via **Upstash Redis** (REST-based, serverless)
- **Session Management** — Multi-user support through session IDs

### Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Framework | LangChain + LCEL |
| Memory | Upstash Redis (cloud) |
| API | FastAPI |

### How to Run

```bash
# From the repo root
uvicorn langchain_chat.main:app --reload
```

### API Endpoint

**`POST /chat`**

```json
{
  "session_id": "user_123",
  "user_query": "Tell me about black holes"
}
```

Response:
```json
{
  "session_id": "user_123",
  "ai_response": "Black holes are regions of spacetime..."
}
```

---

## 🔍 Module 2: RAG Pipeline (`rag_pipeline/`)

A **Retrieval-Augmented Generation** pipeline that grounds LLM answers in real knowledge by first retrieving relevant context from a vector database, then generating an informed response.

### Key Concepts Covered
- **Document Loading & Chunking** — Splitting text documents with `RecursiveCharacterTextSplitter`
- **Embeddings** — Converting text to vectors using HuggingFace (`all-MiniLM-L6-v2`)
- **Vector Storage** — Cloud vector database with **Pinecone**
- **Retrieval-Augmented Generation** — Combining retrieved context with LLM prompting

### Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| Vector DB | Pinecone (cloud, serverless) |
| API | FastAPI |

### How to Run

**Step 1: Ingest documents into Pinecone** (one-time setup)
```bash
python -m rag_pipeline.rag_pipeline
```

**Step 2: Start the RAG API server**
```bash
uvicorn rag_pipeline.rag_fast_api:app --reload
```

### API Endpoint

**`GET /chat?query=your+question+here`**

```
GET /chat?query=What is the Chandrasekhar Limit?
```

Response:
```json
{
  "ai_response": "The Chandrasekhar Limit is exactly 1.44 solar masses...",
  "source_documents": ["knowledge.txt"]
}
```

---

## ⚙️ Environment Setup

### 1. Clone & Install

```bash
git clone https://github.com/just000Curious/Langchain.git
cd Langchain
pip install langchain langchain-core langchain-community langchain-groq langchain-huggingface langchain-pinecone pinecone fastapi uvicorn python-dotenv upstash-redis
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
```

Fill in your API keys in `.env`:

| Variable | Service | Where to Get It |
|---|---|---|
| `GROQ_API_KEY` | Groq | [console.groq.com](https://console.groq.com/) |
| `UPSTASH_REDIS_REST_URL` | Upstash Redis | [console.upstash.com](https://console.upstash.com/) |
| `UPSTASH_REDIS_REST_TOKEN` | Upstash Redis | [console.upstash.com](https://console.upstash.com/) |
| `HUGGINGFACE_API_KEY` | HuggingFace | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `PINECONE_API_KEY` | Pinecone | [app.pinecone.io](https://app.pinecone.io/) |
| `PINECONE_INDEX_NAME` | Pinecone | Your Pinecone dashboard |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                   LangChain Chat Module                 │
│                                                         │
│  User ──▶ FastAPI ──▶ LCEL Chain ──▶ Groq LLM          │
│                          │                              │
│                   Upstash Redis                         │
│                  (Chat Memory)                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                    RAG Pipeline Module                  │
│                                                         │
│  User ──▶ FastAPI ──▶ Embed Query ──▶ Pinecone Search  │
│                                           │             │
│                                    Retrieved Chunks     │
│                                           │             │
│                                     Groq LLM ──▶ Answer│
└─────────────────────────────────────────────────────────┘
```

---

## 📝 License

This project is for educational and learning purposes.
