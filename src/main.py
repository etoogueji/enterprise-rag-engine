import json
import asyncio
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.schemas import QueryRequest, QueryResponse
# Import 'app' from graph.py and alias it as rag_graph
from src.graph import app as rag_graph

fastapi_app = FastAPI(
    title="Enterprise RAG Engine API",
    version="1.0.0",
    description="Production RAG engine featuring Hybrid Search, Re-ranking, and Dynamic LangGraph Routing."
)

# Enable CORS for frontend clients
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@fastapi_app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "healthy", "engine": "Enterprise LangGraph RAG v1.0"}


@fastapi_app.post("/api/v1/rag/query", response_model=QueryResponse)
async def process_rag_query(payload: QueryRequest):
    """Executes the full RAG state graph synchronously and returns complete context and response."""
    try:
        # Matches your GraphState: question, generation, documents, route
        initial_state = {
            "question": payload.query,
            "generation": "",
            "documents": [],
            "route": ""
        }
        
        # Invoke compiled LangGraph workflow asynchronously
        final_state = await rag_graph.ainvoke(initial_state)
        
        # Extract document content strings for context display
        extracted_context = [
            doc.get("parent_text", "") for doc in final_state.get("documents", [])
        ]
        
        return QueryResponse(
            query=payload.query,
            route=final_state.get("route", "UNKNOWN"),
            answer=final_state.get("generation", ""),
            retrieved_context=extracted_context
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG execution error: {str(e)}"
        )


@fastapi_app.post("/api/v1/rag/stream")
async def stream_rag_query(payload: QueryRequest):
    """Streams graph node updates and LLM response tokens in real-time using Server-Sent Events (SSE)."""
    
    async def event_generator():
        initial_state = {
            "question": payload.query,
            "generation": "",
            "documents": [],
            "route": ""
        }
        
        try:
            yield {
                "event": "node_update",
                "data": json.dumps({"status": "Routing query..."})
            }
            
            # Stream graph state execution step-by-step
            async for event in rag_graph.astream_events(initial_state, version="v2"):
                kind = event["event"]
                
                # Signal node execution steps (router, retrieve, generate, direct_response)
                if kind == "on_chain_start" and event["name"] in ["router", "retrieve", "generate", "direct_response"]:
                    yield {
                        "event": "node_execution",
                        "data": json.dumps({"active_node": event["name"]})
                    }
                
                # Stream LLM tokens dynamically as they yield
                elif kind == "on_chat_model_stream":
                    token = event["data"]["chunk"].content
                    if token:
                        yield {
                            "event": "token",
                            "data": json.dumps({"content": token})
                        }
            
            yield {
                "event": "done",
                "data": json.dumps({"status": "completed"})
            }

        except Exception as err:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(err)})
            }

    return EventSourceResponse(event_generator())