import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.graph import app

# 1. Benchmark Dataset
eval_dataset = [
    {
        "user_input": "How do I set up my company email on my mobile device?",
        "reference": "To set up company email on mobile, ensure you have a supported OS (iOS, Android, Windows) and credentials. Open Settings, navigate to Mail/Email, add account as Exchange/Corporate, enter credentials and server settings like mail.company.com, enable SSL/TLS, and verify with a test email."
    },
    {
        "user_input": "What prerequisites are needed before configuring mobile email?",
        "reference": "Prerequisites include a mobile device running a supported operating system (iOS, Android, or Windows), company email account credentials, and installed MDM profile if required by company policy."
    }
]

def run_ragas_evaluation():
    print("[*] Running RAG engine over evaluation questions...")
    
    user_inputs = []
    responses = []
    retrieved_contexts = []
    references = []

    for item in eval_dataset:
        q = item["user_input"]
        gt = item["reference"]
        
        # Invoke LangGraph pipeline
        result = app.invoke({"question": q})
        
        # Extract output and contexts
        generated_answer = result.get("generation", "")
        retrieved_docs = result.get("documents", [])
        extracted_contexts = [doc["parent_text"] for doc in retrieved_docs] if retrieved_docs else [""]
        
        user_inputs.append(q)
        responses.append(generated_answer)
        retrieved_contexts.append(extracted_contexts)
        references.append(gt)

    dataset_dict = {
        "user_input": user_inputs,
        "response": responses,
        "retrieved_contexts": retrieved_contexts,
        "reference": references
    }
    eval_hf_dataset = Dataset.from_dict(dataset_dict)

    print("[*] Initializing Judge LLM (qwen/qwen3.6-27b)...")
    
    # Using Qwen 27B or GPT-OSS-120B on Groq for higher token headroom and fast JSON reasoning
    judge_llm = ChatGroq(
        model="openai/gpt-oss-120b", 
        temperature=0.0,
        max_tokens=4096,
        model_kwargs={"response_format": {"type": "text"}}
    )
    judge_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("[*] Evaluating RAG pipeline across metrics...")

    custom_run_config = RunConfig(
    timeout=300,     # Raise timeout to 5 minutes per metric job
    max_workers=2    # Prevent hitting Groq async rate limits
    )
    
    results = evaluate(
        dataset=eval_hf_dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=judge_llm,
        embeddings = judge_embeddings,
        run_config = custom_run_config,
        raise_exceptions=False
    )

    print("\n================ RAG EVALUATION RESULTS ================")
    print(results)
    print("========================================================\n")
    
    return results

if __name__ == "__main__":
    run_ragas_evaluation()