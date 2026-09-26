import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_groq import ChatGroq

DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
groq_api_key = os.getenv("GROQ_API_KEY")

def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def ingest_pdf(pdf_path: str, filename: str, user_id: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # 1. Ask Groq for a brief identity/summary of the document
    extractor_llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key)
    prompt = f"In one concise sentence, state the type of document this is, who it belongs to, and its core purpose. Text: {pages[0].page_content[:1000]}"
    doc_identity = extractor_llm.invoke(prompt).content
    print(f"Generated Identity for {filename}: {doc_identity}")
    
    # 2. Inject the summary and the user_id into every page
    for page in pages:
        page.metadata['source'] = filename
        page.metadata['user_id'] = user_id  # Secures the chunk to this specific user
        page.page_content = f"Document Identity: {doc_identity}\n\nOriginal Content:\n{page.page_content}"
        
    _chunk_and_store(pages)

def ingest_text(text: str, filename: str, user_id: str):
    # Apply the same logic for images/OCR
    extractor_llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=groq_api_key)
    prompt = f"In one concise sentence, state the type of document this is, who it belongs to, and its core purpose. Text: {text[:1000]}"
    doc_identity = extractor_llm.invoke(prompt).content
    
    augmented_text = f"Document Identity: {doc_identity}\n\nOriginal Content:\n{text}"
    doc = Document(page_content=augmented_text, metadata={"source": filename, "user_id": user_id})
    _chunk_and_store([doc])

def _chunk_and_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB.")