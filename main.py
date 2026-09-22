"""
Salman Physio Care - RAG Chatbot Backend
-----------------------------------------
Ye FastAPI server Colab notebook ka poora RAG pipeline (Qdrant + Cohere + Groq)
production mein chalata hai, aur har website visitor ke liye alag memory rakhta hai.

Local test: uvicorn main:app --reload
Render par deploy hone ke baad, ye ek public URL deta hai jise widget.js call karega.
"""

import os
import uuid
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import init_chat_model
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# ---------------------------------------------------------------------------
# 1. Environment variables (Render ke "Environment" tab mein set karni hain)
# ---------------------------------------------------------------------------
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "salman_physio_docs")

# Website ka domain jahan se widget call karega (Render env var ALLOWED_ORIGINS
# mein comma-separated list dein, jaise: https://salmanphysiocare.com,https://www.salmanphysiocare.com)
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

if not all([COHERE_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise RuntimeError(
        "Missing required environment variables. "
        "Set COHERE_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY on Render."
    )

os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------------------------------------------------------------------------
# 2. Ek dafa startup par: embeddings, vectorstore, aur LLM shuru karna
#    (Colab ke Step 4-6 jaisa, bas yahan ye server shuru hote hi hota hai)
# ---------------------------------------------------------------------------
embeddings = CohereEmbeddings(model="embed-english-v3.0")

vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

llm = init_chat_model("groq:openai/gpt-oss-120b", temperature=0.3, max_tokens=500)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ---------------------------------------------------------------------------
# 3. Har user (session) ki apni alag memory + chain
#    NOTE: Ye memory server ki RAM mein hai. Server restart hone par
#    (Render free tier par ye ho sakta hai) purani sessions ki memory chali jayegi.
#    Bade production scale ke liye isay Redis ya database mein rakhna behtar hoga.
# ---------------------------------------------------------------------------
sessions: dict[str, dict] = {}
SESSION_TIMEOUT = timedelta(hours=2)


def get_or_create_session(session_id: str) -> ConversationalRetrievalChain:
    """Har session_id ke liye alag memory wali chain deta hai."""
    now = datetime.utcnow()

    # Purani, khatam ho chuki sessions saaf karna
    expired = [sid for sid, data in sessions.items() if now - data["last_used"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

    if session_id not in sessions:
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_history",
            max_token_limit=1000,
            return_messages=True,
            output_key="answer",
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            output_key="answer",
        )
        sessions[session_id] = {"chain": chain, "last_used": now}
    else:
        sessions[session_id]["last_used"] = now

    return sessions[session_id]["chain"]


# ---------------------------------------------------------------------------
# 4. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Salman Physio Care Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # Agar nahi diya to naya session banega


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/")
def health_check():
    """Render isay use kar ke check karta hai server zinda hai ya nahi."""
    return {"status": "ok", "service": "Salman Physio Care Chatbot"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message khali nahi ho sakta.")

    session_id = req.session_id or str(uuid.uuid4())
    chain = get_or_create_session(session_id)

    try:
        result = chain.invoke({"question": req.message})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Chatbot mein masla aaya: {exc}")

    return ChatResponse(answer=result["answer"], session_id=session_id)


@app.post("/chat/reset")
def reset_session(session_id: str):
    """User agar naya conversation shuru karna chahe (memory clear karne ke liye)."""
    sessions.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}
