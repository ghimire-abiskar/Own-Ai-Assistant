import os
import shutil
import warnings
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from langchain_groq import ChatGroq
from ocr import extract_text_from_image
from ingest import ingest_pdf, ingest_text, get_vector_store

warnings.filterwarnings("ignore")

app = FastAPI(title="RAG AI Engine Service")

# Initialize Groq LLM
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key)

class QueryRequest(BaseModel):
    question: str

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    print(f"Received request to process: {file.filename}")
    
    # 1. Ensure the upload directory exists
    upload_dir = "/app/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    
    # 2. Save the incoming network file to the Python container's volume
    file_location = os.path.join(upload_dir, file.filename)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
        
    file_ext = file.filename.split('.')[-1].lower()
    
    try:
        # 3. Process the file now that it exists locally on this machine
        if file_ext in ['png', 'jpg', 'jpeg']:
            text = extract_text_from_image(file_location)
            ingest_text(text, file.filename)
            return {"status": "success", "message": f"Image '{file.filename}' OCR complete and added to Vector DB."}
            
        elif file_ext == 'pdf':
            ingest_pdf(file_location, file.filename)
            return {"status": "success", "message": f"PDF '{file.filename}' chunked and added to Vector DB."}
            
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
        print("\n========== RETRIEVED CHUNKS ==========")

        for i, doc in enumerate(docs):
            print(f"\n--- CHUNK {i + 1} ---")
            print("SOURCE:", doc.metadata.get("source"))
            print("PAGE:", doc.metadata.get("page"))
            print("CONTENT:")
            print(doc.page_content)

        print("\n======================================")
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = list(set([doc.metadata.get('source', 'Unknown Document') for doc in docs]))

        # 2. Prompt the LLM
        prompt = f"Answer the question based ONLY on the following context.\n\nContext:\n{context}\n\nQuestion: {request.question}"
        response = llm.invoke(prompt)

        return {"answer": response.content, "sources": sources}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))