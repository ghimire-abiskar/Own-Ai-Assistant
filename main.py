import os
import warnings
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq
from ocr import extract_text_from_image
from ingest import ingest_pdf, ingest_text, get_vector_store

warnings.filterwarnings("ignore")

app = FastAPI(title="RAG AI Engine Service")

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=groq_api_key)

class ProcessRequest(BaseModel):
    filePath: str
    fileName: str

class QueryRequest(BaseModel):
    question: str

@app.post("/process")
async def process_document(request: ProcessRequest):
    print(f"Received request to process: {request.fileName}")
    
    if not os.path.exists(request.filePath):
        raise HTTPException(status_code=404, detail=f"File not found: {request.filePath}")
        
    file_ext = request.fileName.split('.')[-1].lower()
    
    try:
        if file_ext in ['png', 'jpg', 'jpeg']:
            text = extract_text_from_image(request.filePath)
            ingest_text(text, request.fileName)
            return {"status": "success", "message": f"Image '{request.fileName}' OCR complete and added to Vector DB."}
            
        elif file_ext == 'pdf':
            ingest_pdf(request.filePath, request.fileName)
            return {"status": "success", "message": f"PDF '{request.fileName}' chunked and added to Vector DB."}
            
        else:
            return {"status": "ignored", "message": "Unsupported file format."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    try:
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 1. Fetch relevant chunks
        docs = retriever.invoke(request.question)
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get('source', 'Unknown Document') for doc in docs]))

        # 2. Prompt the LLM
        prompt = f"Answer the question based ONLY on the following context.\n\nContext:\n{context}\n\nQuestion: {request.question}"
        response = llm.invoke(prompt)

        return {"answer": response.content, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))