from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Imports the chunking function you just verified in Step 1
from ingest import load_and_chunk_pdf

def create_vector_db(pdf_path: str, db_dir: str):
    print("1. Chunking PDF via ingest.py...")
    chunks = load_and_chunk_pdf(pdf_path)
    
    print("2. Loading Local Embedding Model (downloads ~90MB model on first run)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    print("3. Generating embeddings & storing vectors in ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    print(f"\nSuccess! Local vector database saved at: {db_dir}")
    return vector_store

if __name__ == "__main__":
    create_vector_db("sample.pdf", "./chroma_db")