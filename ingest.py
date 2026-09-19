from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

DB_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_vector_store():
    # Downloads the ~90MB model automatically on first run
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

def ingest_pdf(pdf_path: str, filename: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    # Update metadata to include the filename for source tracking
    for page in pages:
        page.metadata['source'] = filename
        
    _chunk_and_store(pages)

def ingest_text(text: str, filename: str):
    # Wrap raw OCR text into a LangChain Document
    doc = Document(page_content=text, metadata={"source": filename})
    _chunk_and_store([doc])

def _chunk_and_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
    
    vector_store = get_vector_store()
    vector_store.add_documents(chunks)
    print(f"Successfully ingested {len(chunks)} chunks into ChromaDB.")