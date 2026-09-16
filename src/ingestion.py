import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
COLLECTION_NAME = "enterprise_docs"

def initialize_qdrant_collection(client: QdrantClient, vector_size: int = 384):
    """Ensures the Qdrant collection exists with the exact vector dimension required."""
    collections = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME in collections:
        # Check existing vector size
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        current_size = collection_info.config.params.vectors.size
        
        if current_size != vector_size:
            print(f"[*] Recreating collection '{COLLECTION_NAME}' (updating dimension from {current_size} to {vector_size})...")
            client.delete_collection(collection_name=COLLECTION_NAME)
            collections.remove(COLLECTION_NAME)

    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        print(f"[+] Collection '{COLLECTION_NAME}' created with dimension {vector_size}.")

def process_and_index_pdf(pdf_path: str):
    """Loads a PDF, creates parent-child chunks, embeds locally, and indexes to Qdrant."""
    print(f"[*] Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    parent_docs = parent_splitter.split_documents(documents)

    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    
    print("[*] Loading local HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    qdrant_kwargs = {"url": QDRANT_URL}
    if QDRANT_API_KEY:
        qdrant_kwargs["api_key"] = QDRANT_API_KEY

    qdrant = QdrantClient(**qdrant_kwargs)
    
    # Force 384-dim check
    initialize_qdrant_collection(qdrant, vector_size=384)

    points = []
    point_id = 1

    print("[*] Generating chunk embeddings locally and preparing payloads...")
    for p_idx, parent in enumerate(parent_docs):
        child_docs = child_splitter.split_documents([parent])
        for child in child_docs:
            vector = embeddings.embed_query(child.page_content)
            
            payload = {
                "child_text": child.page_content,
                "parent_text": parent.page_content,
                "page": child.metadata.get("page", 0),
                "source": os.path.basename(pdf_path),
                "parent_id": p_idx
            }
            
            points.append({
                "id": point_id,
                "vector": vector,
                "payload": payload
            })
            point_id += 1

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"[+] Successfully indexed {len(points)} child chunks into Qdrant.")

if __name__ == "__main__":
    sample_pdf = "data/sample_contracts.pdf"
    if os.path.exists(sample_pdf):
        process_and_index_pdf(sample_pdf)
    else:
        print(f"[!] Place a valid PDF at '{sample_pdf}' to run ingestion.")