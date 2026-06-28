from fastapi import FastAPI
from rag_pipeline.rag_llm import CloudRAGPipeline

app = FastAPI()

rag_service = CloudRAGPipeline()


@app.get("/chat")
def chat(query:str):
    result = rag_service.query(query)
    return {"ai_response":result["answer"], "source_documents":[doc.metadata.get("source") for doc in result["context"]]}
