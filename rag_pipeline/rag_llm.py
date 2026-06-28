import os
from dotenv import load_dotenv
load_dotenv()

from pinecone import Pinecone as PineconeClient
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


class CloudRAGPipeline:
    """
    A unified class for the RAG pipeline.
    This allows FastAPI to instantiate the embeddings, database connection,
    and LLM only once on startup, preventing slow requests.

    Uses the pinecone SDK directly (instead of langchain-pinecone) for
    Python 3.14 compatibility.
    """
    def __init__(self):
        print("Initializing Cloud RAG Pipeline (Loading models & connecting to Pinecone)...")
        # Load heavy embeddings and connections only ONCE during initialization
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Connect to Pinecone directly via the official SDK
        pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

        self.model = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9)

        system_prompt = """Answer the question based on the context provided below. 
        If the answer is not in the context, say "I don't have information on that."
        Context: {context}"""

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}")
            ]
        )
        print("Pipeline initialized successfully!")

    def query(self, user_query: str, top_k: int = 2):
        print(f"\nExecuting cloud-augmented retrieval for: {user_query}")

        # Embed the query
        query_vector = self.embeddings.embed_query(user_query)

        # Query Pinecone directly
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        # Convert Pinecone results to LangChain Documents
        retrieved_docs = []
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            page_content = metadata.pop("text", "") or metadata.pop("page_content", "")
            retrieved_docs.append(Document(page_content=page_content, metadata=metadata))

        context_str = "\n\n".join(doc.page_content for doc in retrieved_docs)

        # Generate the answer using Groq
        prompt_value = self.prompt.invoke({"context": context_str, "input": user_query})
        ai_msg = self.model.invoke(prompt_value)

        return {
            "answer": ai_msg.content,
            "context": retrieved_docs
        }


if __name__ == "__main__":
    # Test the class functionality
    rag_service = CloudRAGPipeline()

    test_query = "what is the chandrasekhar limit?"
    result = rag_service.query(test_query)

    print("\nAI Answer:")
    print(f"{result['answer']}")

    print(f"\nChunks utilized:")
    for doc in result["context"]:
        print(f" _ from : {doc.metadata.get('source', 'Unknown')}")
        print("-" * 60)
