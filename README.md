# Enterprise RAG with Hybrid Search & Evaluation Engine

## Document intelligence system for specific domains (e.g., medical guidelines, legal contracts, or technical documentation) which handles complex queries with GUARANTEED accuracy

### Key Architecture & Technical Stack:
- __Ingestion & Processing__: Pandas is used for heavy PDF/unstructured textual datasets, breaking them down into chunking hierarchies (parent-child chunking)

- __Hybrid Search Retrieval__: A dual-retrieval pipeline combining __Dense Search__ (vector embeddings using Qdrant or Pinecone) and Sparse Search (BM25 via Elasticsearch/Typesense) re-ranked using a Cross-Encoder

- __LLM Orchestration__: LangGraph to manage cyclic agent state transitions (handling fallback strategies when retrieval confidence is low)

- __Evaluation__: An automated evaluation pipeline using Ragas or Trulens to measure _Faithfulness_, *Answer Relevance*, and _Context Precision_

### In summary, instead of just returning basic vector search results, it will route complex queries, combine dense vector embeddings with sparse keyword search (BM25), re-rank the retrieved passages, and run continuous evaluation to prove zero hallucination (incorrect/fabricated output from an LLM)
