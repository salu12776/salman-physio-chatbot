"""
Salman Physio Care - RAG Chatbot Backend
-----------------------------------------
Ye FastAPI server Colab notebook ka poora RAG pipeline (Qdrant + Cohere + Groq)
production mein chalata hai, aur har website visitor ke liye alag memory rakhta hai.

Local test: uvicorn main:app --reload
Render par deploy hone ke baad, ye ek public URL deta hai jise widget call karega.

Security features:
- API keys sirf environment variables se (code mein kabhi nahi)
- CORS sirf allowed website ke liye
- Rate limiting: har user (IP) aur poore server ki had
- Message length ki had
- Error details user ko nahi, sirf Render logs mein
- Medical safety guardrails prompt mein
"""

import os
import uuid
import logging
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import init_chat_model
from langchain_classic.memory import ConversationSummaryBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# Errors Render ke "Logs" tab mein nazar aayenge
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physio-chatbot")

# ---------------------------------------------------------------------------
# 1. Environment variables (Render ke "Environment" tab mein set karni hain)
# ---------------------------------------------------------------------------
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "salman_physio_docs")

# Default ab "*" nahi, balke sirf GitHub Pages domain hai.
# Env var na bhi ho to backend safe rahega.
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "https://salu12776.github.io").split(",")
    if origin.strip()
]

if not all([COHERE_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY]):
    raise RuntimeError(
        "Missing required environment variables. "
        "Set COHERE_API_KEY, GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY on Render."
    )

os.environ["COHERE_API_KEY"] = COHERE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# ---------------------------------------------------------------------------
# 2. Ek dafa startup par: embeddings, vectorstore, aur LLM shuru karna
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

# Prompt: sirf context se jawab, Roman Urdu/English, aur medical safety
CUSTOM_PROMPT = PromptTemplate(
    template="""You are a helpful assistant for Salman Physio Care, a physiotherapy clinic.
Answer the question using only the context below.
Always reply in Roman Urdu or English, written in the Latin/English alphabet only.
Never use Devanagari, Arabic, or any other script.
Do not use markdown formatting. Never use ** or * or # symbols.
Write in plain text only. For lists, start each item on a new line with a dash (-).
If you don't know the answer from the context, say so honestly and suggest calling the clinic at 0325-9874794.

Medical safety rules (always follow these):
- Never diagnose any condition and never suggest or name medicines.
- Only share general information about the clinic's services, fees, timings and booking.
- If the user mentions severe or sudden pain, chest pain, numbness, weakness in arms or legs,
  loss of bladder or bowel control, a recent accident or fall, or a high fever, tell them to
  see a doctor or go to the emergency department immediately, before anything else.
- For any specific health concern, encourage them to book an assessment with the physiotherapist.

Context: {context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)

# ---------------------------------------------------------------------------
# 3. Har user (session) ki apni alag memory + chain
# NOTE: Memory server ki RAM mein hai; restart par purani sessions chali jayengi.
# ---------------------------------------------------------------------------
sessions: dict[str, dict] = {}
SESSION_TIMEOUT = timedelta(hours=2)
MAX_SESSIONS = 300  # RAM bhar jane se bachane ke liye had


def get_or_create_session(session_id: str) -> ConversationalRetrievalChain:
    """Har session_id ke liye alag memory wali chain deta hai."""
    now = datetime.now(timezone.utc)

    # Purani, khatam ho chuki sessions saaf karna
    expired = [sid for sid, data in sessions.items() if now - data["last_used"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

    if session_id not in sessions:
        # Had poori ho to sab se purani session hata dein
        if len(sessions) >= MAX_SESSIONS:
            oldest = min(sessions, key=lambda sid: sessions[sid]["last_used"])
            del sessions[oldest]

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
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        )
        sessions[session_id] = {"chain": chain, "last_used": now}
    else:
        sessions[session_id]["last_used"] = now

    return sessions[session_id]["chain"]


# ---------------------------------------------------------------------------
# 4. Rate limiting
# Render ek proxy ke peeche chalta hai, is liye asal user ka IP
# "X-Forwarded-For" header se lete hain.
# ---------------------------------------------------------------------------
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def global_key(request: Request) -> str:
    """Poore server ke liye ek hi counter (API credits bachane ke liye)."""
    return "global"


limiter = Limiter(key_func=get_client_ip)

# ---------------------------------------------------------------------------
# 5. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Salman Physio Care Chatbot API")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Aap ne bohat zyada messages bhej diye hain. "
                      "Thori der baad dobara try karein, ya clinic ko 0325-9874794 par call karein."
        },
    )


app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)
    session_id: str | None = Field(None, max_length=64)


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.get("/")
def health_check():
    """Render aur UptimeRobot isay check karte hain ke server zinda hai. Is par limit nahi."""
    return {"status": "ok", "service": "Salman Physio Care Chatbot"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("15/minute")                            # har user: 15 messages per minute
@limiter.limit("100/day")                              # har user: 100 messages per din
@limiter.limit("500/hour", key_func=global_key)        # poora server: 500 messages per ghanta
def chat(request: Request, req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message khali nahi ho sakta.")

    session_id = req.session_id or str(uuid.uuid4())
    chain = get_or_create_session(session_id)

    try:
        result = chain.invoke({"question": req.message.strip()})
    except Exception:
        # Poori error sirf Render logs mein, user ko aam sa message
        logger.exception("Chat chain failed for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail="Maazrat, abhi masla hai. Thori der baad try karein ya clinic ko 0325-9874794 par call karein.",
        )

    return ChatResponse(answer=result["answer"], session_id=session_id)


@app.post("/chat/reset")
@limiter.limit("10/minute")
def reset_session(request: Request, session_id: str):
    """User naya conversation shuru karna chahe to memory clear karna."""
    sessions.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}
