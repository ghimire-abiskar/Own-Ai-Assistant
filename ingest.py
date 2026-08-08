from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk_pdf(pdf_path: str):
    print(f"Loading PDF: {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    print(f"Successfully loaded {len(pages)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(pages)
    print(f"Total chunks created: {len(chunks)}\n")
    return chunks

if __name__ == "__main__":
    sample_pdf = "sample.pdf" 
    try:
        chunks = load_and_chunk_pdf(sample_pdf)
        print("=== Sample Chunk Preview ===")
        print(f"Page Number: {chunks[0].metadata['page']}")
        print(f"Content:\n{chunks[0].page_content[:200]}...")
    except Exception as e:
        print(f"Error: {e}")