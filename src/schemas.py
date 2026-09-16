# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str = Field(..., example="How do I set up my company email on my mobile device?")
    session_id: Optional[str] = Field(default="default_session", example="user_123")

class DocumentChunk(BaseModel):
    content: str
    score: float
    source: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    route: str
    answer: str
    retrieved_context: List[str] = []