import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Use UpstashRedisChatMessageHistory for REST-based Upstash DB
from langchain_community.chat_message_histories import UpstashRedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

# Import the decoupled LangChain logic from our other file
from langchain_chat.llm_chain import create_base_chain

load_dotenv()

chain_with_memory = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global chain_with_memory
    
    # 1. Load Upstash credentials
    upstash_url = os.getenv("UPSTASH_REDIS_REST_URL")
    upstash_token = os.getenv("UPSTASH_REDIS_REST_TOKEN")
    
    # 2. LangChain Logic: Fetch the base chain assembly line
    base_chain = create_base_chain()
    
    # 3. Integration: Wrap the base chain with Upstash Redis memory history
    def get_session_history(session_id: str):
        # UpstashRedisChatMessageHistory uses the REST API natively!
        return UpstashRedisChatMessageHistory(
            session_id=session_id, 
            url=upstash_url, 
            token=upstash_token
        )
    
    chain_with_memory = RunnableWithMessageHistory(
        base_chain,
        get_session_history,
        input_messages_key="user_query",
        history_messages_key="chat_history"
    )
    
    print("Cloud Redis Memory Initialized!")
    yield  # ⏸️ Server runs here
    print("Server Shutting Down...")

app = FastAPI(lifespan=lifespan)

class ChatRequest(BaseModel):
    session_id: str
    user_query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        config = {"configurable": {"session_id": request.session_id}}
        response = chain_with_memory.invoke(
            {"user_query": request.user_query}, 
            config=config
        )
        return {"session_id": request.session_id, "ai_response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
