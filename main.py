import os
import shutil
import warnings
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from langchain_groq import ChatGroq

from ocr import extract_text_from_image
from ingest import ingest_pdf, ingest_text, get_vector_store

# Use these stable imports matching your requirements.txt:
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

warnings.filterwarnings("ignore")

app = FastAPI(title="RAG AI Engine Service")

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key)

# The query request now strictly requires the user_id
class QueryRequest(BaseModel):
    question: str
    user_id: str

@app.post("/process")
async def process_document(
    file: UploadFile = File(...), 
    user_id: str = Form(...)  # Accepts the user_id alongside the multipart file
):
    print(f"Received request to process: {file.filename} for user: {user_id}")
    
    upload_dir = "/app/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    file_ext = file.filename.split('.')[-1].lower()
    
    try:
        if file_ext in ['png', 'jpg', 'jpeg']:
            text = extract_text_from_image(file_location)
            ingest_text(text, file.filename, user_id)
            return {"status": "success", "message": f"Image '{file.filename}' processed for {user_id}."}
            
        elif file_ext == 'pdf':
            ingest_pdf(file_location, file.filename, user_id)
            return {"status": "success", "message": f"PDF '{file.filename}' processed for {user_id}."}
            
        else:
            return {"status": "ignored", "message": "Unsupported file format."}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def handle_query(request: QueryRequest):
    try:
        vector_store = get_vector_store()
        
        # 1. Base Retrieval: Fetch 15 chunks, strictly locked to this user_id
        base_retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 15,
                "filter": {"user_id": request.user_id}
            }
        )
        
        # 2. Re-Ranking: Use the Cross-Encoder to score the 15 chunks and keep the top 4
        model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
        compressor = CrossEncoderReranker(model=model, top_n=4)
        
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=base_retriever
        )
        
        # 3. Execute the search
        docs = compression_retriever.invoke(request.question)
        
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get('source', 'Unknown Document') for doc in docs]))

        # 4. Generate the final answer
        prompt = f"Answer the question based ONLY on the following context.\n\nContext:\n{context}\n\nQuestion: {request.question}"
        response = await llm.ainvoke(prompt)

        return {"answer": response.content, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))