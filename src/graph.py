import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from src.retriever import HybridRerankRetriever

load_dotenv()

# initializing the LLM via Groq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0
)

# Initialize Retriever
retriever = HybridRerankRetriever(top_k=3)

# Define LangGraph State
class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[dict]
    route: str

# Node 1: Router
def route_question(state: GraphState):
    """Determines whether to route to RAG retrieval or direct response."""
    question = state["question"]
    
    prompt = f"""You are an enterprise routing agent. Analyze the input question:
'{question}'

Determine if this query requires retrieving specific internal knowledge/documents (e.g., policies, contract terms, technical setups) OR if it is a general greeting/chitchat (e.g., "hi", "how are you").

Respond with EXACTLY one word: 'RETRIEVE' or 'DIRECT'."""

    response = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    
    route = "RETRIEVE" if "RETRIEVE" in response else "DIRECT"
    print(f"[*] Decision: Route query to -> {route}")
    return {"route": route}

# Node 2: Retrieve
def retrieve_node(state: GraphState):
    """Executes Hybrid Search + Re-ranking pipeline."""
    question = state["question"]
    docs = retriever.search(question)
    return {"documents": docs}

# Node 3: Generate Answer
def generate_node(state: GraphState):
    """Generates ground-truth answer using retrieved context."""
    question = state["question"]
    docs = state.get("documents", [])

    context = "\n\n".join([f"--- Excerpt {idx+1} ---\n{doc['parent_text']}" for idx, doc in enumerate(docs)])

    prompt = f"""You are a precise enterprise AI assistant. Answer the user question using ONLY the provided context excerpts below. 
If the answer cannot be determined from the context, state that you do not have enough information.

Context:
{context}

Question: {question}
Answer:"""

    response = llm.invoke([HumanMessage(content=prompt)]).content
    return {"generation": response}

# Node 4: Direct Response
def direct_response_node(state: GraphState):
    """Handles general conversational queries directly."""
    question = state["question"]
    response = llm.invoke([
        SystemMessage(content="You are a helpful enterprise AI collaborator. Respond politely to general questions or greetings."),
        HumanMessage(content=question)
    ]).content
    return {"generation": response}

# Conditional Routing Logic
def decide_next_node(state: GraphState):
    if state["route"] == "RETRIEVE":
        return "retrieve"
    return "direct_response"

# Build LangGraph State Machine
workflow = StateGraph(GraphState)

workflow.add_node("router", route_question)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("direct_response", direct_response_node)

# Set Entry Point & Edges
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "retrieve": "retrieve",
        "direct_response": "direct_response"
    }
)

workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
workflow.add_edge("direct_response", END)

# Compile Graph
app = workflow.compile()

if __name__ == "__main__":
    print("\n--- Test 1: Technical Query ---")
    res1 = app.invoke({"question": "How do I set up my company email on my mobile device?"})
    print(f"\nResponse:\n{res1['generation']}\n")

    print("\n--- Test 2: Conversational Query ---")
    res2 = app.invoke({"question": "Hey! How are you doing today?"})
    print(f"\nResponse:\n{res2['generation']}\n")