import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "enterprise_docs"

class HybridRerankRetriever:
    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        print("[*] Initializing local embedding model (all-MiniLM-L6-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        print("[*] Loading local Cross-Encoder re-ranker (bge-reranker-base)...")
        # Downloads free lightweight local cross-encoder model
        self.reranker = CrossEncoder("BAAI/bge-reranker-base")

        qdrant_kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            qdrant_kwargs["api_key"] = QDRANT_API_KEY
            
        self.qdrant = QdrantClient(**qdrant_kwargs)
        self.bm25 = None
        self.documents_store = []
        
        self._build_bm25_index()

    def _build_bm25_index(self):
        """Fetches ALL indexed payloads from Qdrant using pagination to construct the BM25 index."""
        print("[*] Fetching all documents from Qdrant to construct BM25 index...")
        
        all_records = []
        next_offset = None
        
        # Paginate through all collections in batches
        while True:
            records, next_offset = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=100,  # Batch size
                offset=next_offset,
                with_payload=True,
                with_vectors=False
            )
            all_records.extend(records)
            
            # If next_offset is None, we've retrieved all points
            if next_offset is None:
                break
        
        self.documents_store = [record.payload for record in all_records]
        corpus = [doc["child_text"].lower().split() for doc in self.documents_store]
        self.bm25 = BM25Okapi(corpus)
        print(f"[+] BM25 Index successfully initialized across ALL {len(self.documents_store)} chunks.")

    def dense_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Executes Dense Vector Search via Qdrant."""
        query_vector = self.embeddings.embed_query(query)
        search_results = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points
        return [hit.payload for hit in search_results]

    def sparse_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Executes Sparse Keyword Search via BM25."""
        tokenized_query = query.lower().split()
        return self.bm25.get_top_n(tokenized_query, self.documents_store, n=limit)

    def reciprocal_rank_fusion(
        self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60
    ) -> List[Dict[str, Any]]:
        """Combines Dense and Sparse ranks using RRF."""
        rrf_scores = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["child_text"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (k + (rank + 1))

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["child_text"]
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = {"doc": doc, "score": 0.0}
            rrf_scores[doc_id]["score"] += 1.0 / (k + (rank + 1))

        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:10]] # Pass top 10 candidates to Cross-Encoder

    def re_rank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Applies Cross-Encoder Re-Ranking on candidate documents."""
        if not candidates:
            return []

        # Prepare (query, passage) pairs for Cross-Encoder
        pairs = [[query, doc["parent_text"]] for doc in candidates]
        
        # Calculate cross-encoder relevance scores
        scores = self.reranker.predict(pairs)

        # Attach scores to documents
        for idx, score in enumerate(scores):
            candidates[idx]["rerank_score"] = float(score)

        # Sort documents by re-ranking score descending
        ranked_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked_candidates[:self.top_k]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """Full Hybrid Search Pipeline: Dense + Sparse -> RRF -> Cross-Encoder Re-Rank."""
        print(f"\n[*] Executing Hybrid Search for query: '{query}'")
        dense_hits = self.dense_search(query, limit=10)
        sparse_hits = self.sparse_search(query, limit=10)
        
        fused_candidates = self.reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        print("[*] Re-ranking candidate passages with BGE-Cross-Encoder...")
        final_results = self.re_rank(query, fused_candidates)
        return final_results

if __name__ == "__main__":
    retriever = HybridRerankRetriever(top_k=3)
    
    test_query = "How do I set up my company email on my mobile device?"
    results = retriever.search(test_query)
    
    print("\n--- FINAL RE-RANKED TOP RESULTS ---")
    for idx, doc in enumerate(results, 1):
        print(f"\n[Rank {idx}] (Re-Rank Score: {doc['rerank_score']:.4f})")
        print(f"Source: {doc['source']}, Page: {doc['page']}")
        print(f"Context Excerpt: {doc['parent_text'][:200]}...")