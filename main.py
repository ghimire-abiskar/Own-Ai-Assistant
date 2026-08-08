import os
import warnings
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

app = FastAPI(title="RAG AI Engine Service")

# Request / Response Schemas
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]

# Global references for AI pipeline components
retriever = None
llm = None

@app.on_event("startup")
def startup_event():
    global retriever, llm
    
    # 1. Grab API key from Render Environment Variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("WARNING: GROQ_API_KEY environment variable not set!")

    print("⚡ Loading Embedding Model & connecting to ChromaDB...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print("⚡ Connecting to Groq LLM Service...")
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)
    print("✅ RAG Microservice initialization complete!")

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "python-rag-microservice"}

@app.post("/api/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in docs]))

        prompt = f"Answer the question based ONLY on the following context.\n\nContext:\n{context}\n\nQuestion: {request.question}"
        response = llm.invoke(prompt)

        return QueryResponse(answer=response.content, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")