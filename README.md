# Enterprise RAG Engine with Dynamic Routing & Hybrid Search

A production-grade Document Intelligence System featuring Hybrid Vector/Sparse Retrieval, Cross-Encoder Re-ranking, and dynamic agentic routing managed by LangGraph.

## System Architecture

```text
[ User Query ]
       │
       ▼
[ LangGraph Router Node ] ────► (General Conversation) ──► [ Direct LLM Response ]
       │
       ▼ (Technical / Enterprise Query)
[ Hybrid Search Engine ]
   ├── Dense Retrieval (Qdrant Vector DB / all-MiniLM-L6-v2)
   └── Sparse Retrieval (BM25 Lexical Keyword Matching)
       │
       ▼
[ Reciprocal Rank Fusion (RRF) ]
       │
       ▼
[ Cross-Encoder Re-Ranker (bge-reranker-base) ]
       │
       ▼
[ Parent Context Assembler ] ──► [ LLM Synthesis (Groq Llama-3) ] ──► [ Output ]