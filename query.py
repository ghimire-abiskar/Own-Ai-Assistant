import os
import warnings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore")

# 1. Add your Groq API Key here
os.environ["GROQ_API_KEY"] = "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"

def setup_chat_engine(db_dir: str):
    # Load the exact same embedding model used in Step 2
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Connect to your existing local database
    vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    
    # Connect to the Llama 3 model via Groq's fast inference cloud
    llm = ChatGroq(model_name="llama-3.3-70b-versatile")
    
    # Turn the vector store into a retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    return retriever, llm

if __name__ == "__main__":
    print("Connecting to database and Groq API...")
    retriever, llm = setup_chat_engine("./chroma_db")
    
    print("\n✅ RAG System Ready with Source Citations! (Type 'exit' to quit)\n")
    
    while True:
        user_question = input("Ask a question about your PDF: ")
        if user_question.lower() == 'exit':
            break
            
        print("Searching database & thinking...")
        
        # 1. Retrieve relevant chunks from ChromaDB
        docs = retriever.invoke(user_question)
        
        # 2. Extract context and format page numbers
        context = "\n\n".join([doc.page_content for doc in docs])
        # Page numbers in PyPDF metadata are 0-indexed, so we add 1 for readability
        sources = set([f"Page {doc.metadata.get('page', 0) + 1}" for doc in docs]) 
        
        # 3. Create the explicit instruction prompt for the LLM
        prompt = f"""Answer the question based ONLY on the following context.
        
Context:
{context}

Question: {user_question}
"""
        
        # 4. Generate and print the final answer
        response = llm.invoke(prompt)
        
        print(f"\n🤖 Answer: {response.content}\n")
        print(f"📌 Sources Used: {', '.join(sources)}")
        print("-" * 50)