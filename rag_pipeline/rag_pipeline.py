import os
from dotenv import load_dotenv
load_dotenv()   

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Resolve the knowledge file path relative to this script's directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_KNOWLEDGE_FILE = os.path.join(_SCRIPT_DIR, "knowledge.txt")

def build_cloud_vector_db(file_path: str):
    print("Uploading knowledge base to Pinecone Cloud...")
    loader = TextLoader(file_path, encoding="utf-8")
    raw_docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(raw_docs)
    
    # This pushes the documents directly to your Pinecone index
    vector_db = Pinecone.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print("Cloud Vector DB Built Successfully!")
    return vector_db


if __name__ == "__main__":
    # 1. Upload documents to the cloud (we do this once to populate the empty index)
    build_cloud_vector_db(_KNOWLEDGE_FILE)
    
    # 2. Connect to the cloud vector db to query it
    print("Connecting to cloud vector db (Pinecone)...........")
    vector_db = Pinecone(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings)
    
    retriever = vector_db.as_retriever(search_kwargs={"k": 2})

    # 3. Test the retriever
    query = "What is the Chandrasekhar Limit?"
    print(f"Searching for relevant documents for the query: {query}")

    matched_docs = retriever.invoke(query)
    print(f"Found {len(matched_docs)} relevant documents")

    for i,doc in enumerate(matched_docs,1):
        print(f"\n--- Document {i} ---")
        print(doc.page_content)
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
