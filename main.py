"""
Salman Physio Care - Agentic RAG Chatbot Backend
-------------------------------------------------
Ye FastAPI server ek AI agent chalata hai jiske paas 3 tools hain:
  1. search_clinic_info  -> Qdrant + Cohere RAG (fees, services, timings)
  2. check_availability  -> Google Sheet se khali slots
  3. book_appointment    -> Google Sheet mein nayi booking

Local test: uvicorn main:app --reload

Security features:
- API keys aur Google credentials sirf Render environment / secret files mein
- CORS sirf allowed website ke liye
- Rate limiting: har user (IP) aur poore server ki had
- Message length ki had, har session mein max bookings ki had
- Sheet mein RAW likhna (formula injection se bachao)
- Error details user ko nahi, sirf Render logs mein
- Medical safety guardrails system prompt mein
"""

import os
import re
import uuid
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from slowapi import Limiter
from slowapi.errors import RateLimitExceeded

import gspread
from langchain_cohere import CohereEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("physio-chatbot")

# ---------------------------------------------------------------------------
# 1. Environment variables
# ---------------------------------------------------------------------------
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "salman_physio_docs")

# Google Sheet (Render: SHEET_ID env var + Secret File google-credentials.json)
SHEET_ID = os.environ.get("SHEET_ID", "").strip()
GOOGLE_CREDS_FILE = os.environ.get("GOOGLE_CREDENTIALS_FILE", "/etc/secrets/google-credentials.json")

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
# 2. Clinic ke booking rules
# ---------------------------------------------------------------------------
PKT = ZoneInfo("Asia/Karachi")
CLINIC_PHONE = "0325-9874794"
OPEN_HOUR, CLOSE_HOUR = 10, 20                                   # 10 AM - 8 PM
SLOT_TIMES = [f"{h:02d}:00" for h in range(OPEN_HOUR, CLOSE_HOUR)]  # 10:00 ... 19:00
SLOT_CAPACITY = 2              # 2 physiotherapists, is liye ek slot mein 2 bookings
BOOKING_DAYS_AHEAD = 30        # zyada se zyada 30 din aage tak booking
MAX_BOOKINGS_PER_SESSION = 2   # spam se bachao

SERVICES = {
    "Back Pain Treatment": ["back", "kamar"],
    "Sports Injury Treatment": ["sport", "ligament", "strain"],
    "Post-Surgery Rehabilitation": ["surgery", "rehab", "operation"],
    "Neck & Shoulder Pain": ["neck", "shoulder", "gardan", "kandha"],
}

# ---------------------------------------------------------------------------
# 3. RAG + LLM (startup par ek dafa)
# ---------------------------------------------------------------------------
embeddings = CohereEmbeddings(model="embed-english-v3.0")
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=COLLECTION_NAME,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = init_chat_model("groq:openai/gpt-oss-120b", temperature=0.2, max_tokens=1000)

# ---------------------------------------------------------------------------
# 4. Google Sheet helpers
# Sheet columns: Booking ID | Name | Phone | Service | Date | Time | Created At | Status
# ---------------------------------------------------------------------------
_worksheet = None


def get_worksheet():
    """Sheet se connection sirf pehli zaroorat par banta hai."""
    global _worksheet
    if _worksheet is None:
        if not SHEET_ID or not os.path.exists(GOOGLE_CREDS_FILE):
            raise RuntimeError("Booking sheet configure nahi hai (SHEET_ID ya credentials file missing).")
        client = gspread.service_account(filename=GOOGLE_CREDS_FILE)
        _worksheet = client.open_by_key(SHEET_ID).sheet1
    return _worksheet


def get_active_bookings(date_str: str) -> list[list[str]]:
    """Ek din ki sari bookings jo cancel nahi hui."""
    rows = get_worksheet().get_all_values()[1:]  # pehli row headings hai
    return [
        r for r in rows
        if len(r) >= 8 and r[4].strip() == date_str and r[7].strip().lower() != "cancelled"
    ]


def validate_date(date_str: str):
    """Date check karta hai. Returns (date, None) ya (None, error message)."""
    try:
        d = datetime.strptime(date_str.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None, "Date ka format YYYY-MM-DD hona chahiye, jaise 2026-09-28."
    today = datetime.now(PKT).date()
    if d < today:
        return None, "Ye date guzar chuki hai. Aaj ya aane wali koi date chunein."
    if d > today + timedelta(days=BOOKING_DAYS_AHEAD):
        return None, f"Sirf agle {BOOKING_DAYS_AHEAD} din tak ki booking ho sakti hai."
    if d.weekday() == 6:
        return None, "Sunday ko clinic band hota hai. Monday se Saturday mein koi din chunein."
    return d, None


def get_free_slots(d) -> list[str]:
    counts: dict[str, int] = {}
    for row in get_active_bookings(d.isoformat()):
        counts[row[5].strip()] = counts.get(row[5].strip(), 0) + 1

    now = datetime.now(PKT)
    free = []
    for t in SLOT_TIMES:
        if counts.get(t, 0) >= SLOT_CAPACITY:
            continue
        if d == now.date() and int(t[:2]) <= now.hour:  # aaj ke guzre hue slots
            continue
        free.append(t)
    return free


def match_service(text: str) -> str | None:
    text = text.strip().lower()
    for name, keywords in SERVICES.items():
        if text == name.lower() or any(k in text for k in keywords):
            return name
    return None


def normalize_phone(phone: str) -> str | None:
    digits = re.sub(r"[^\d+]", "", phone)
    if digits.startswith("+92"):
        digits = "0" + digits[3:]
    elif digits.startswith("92") and len(digits) == 12:
        digits = "0" + digits[2:]
    return digits if re.fullmatch(r"03\d{9}", digits) else None


def slot_label(t: str) -> str:
    """'15:00' -> '3:00 PM'"""
    return datetime.strptime(t, "%H:%M").strftime("%I:%M %p").lstrip("0")


# ---------------------------------------------------------------------------
# 5. Agent ke tools (har session ke liye banaye jate hain)
# ---------------------------------------------------------------------------
def make_tools(session: dict):

    @tool
    def search_clinic_info(query: str) -> str:
        """Search the clinic's knowledge base for services, fees, packages, timings,
        staff, home visits, payment and cancellation policy. Use this for ANY
        question about the clinic instead of answering from memory."""
        docs = retriever.invoke(query)
        if not docs:
            return "Clinic ki maloomat mein is bare mein kuch nahi mila."
        return "\n\n".join(doc.page_content for doc in docs)

    @tool
    def check_availability(date: str) -> str:
        """Check free appointment slots on a date. `date` must be YYYY-MM-DD.
        Always call this before booking."""
        d, error = validate_date(date)
        if error:
            return error
        try:
            slots = get_free_slots(d)
        except Exception:
            logger.exception("Availability check failed")
            return f"Booking system abhi available nahi. Clinic ko {CLINIC_PHONE} par call karein."
        day = d.strftime("%A, %d %B %Y")
        if not slots:
            return f"{day} ko koi slot khali nahi. Koi aur din try karein."
        labels = ", ".join(f"{slot_label(t)} ({t})" for t in slots)
        return f"{day} ko ye slots khali hain: {labels}"

    @tool
    def book_appointment(name: str, phone: str, service: str, date: str, time: str) -> str:
        """Book an appointment. ONLY call this after the user has explicitly confirmed
        all the details. `phone`: Pakistani mobile number like 03001234567.
        `service`: one of Back Pain Treatment, Sports Injury Treatment,
        Post-Surgery Rehabilitation, Neck & Shoulder Pain.
        `date`: YYYY-MM-DD. `time`: HH:MM in 24-hour format, from check_availability."""
        if session["bookings"] >= MAX_BOOKINGS_PER_SESSION:
            return f"Is chat se zyada bookings nahi ho sakti. Mazeed booking ke liye {CLINIC_PHONE} par call karein."

        name = name.strip()
        if not (2 <= len(name) <= 60):
            return "Meharbani karke apna sahi naam batayein."

        phone_clean = normalize_phone(phone)
        if not phone_clean:
            return "Phone number sahi nahi. Pakistani mobile number dein, jaise 03001234567."

        service_name = match_service(service)
        if not service_name:
            return "Service samajh nahi aayi. Options: " + ", ".join(SERVICES)

        d, error = validate_date(date)
        if error:
            return error

        time = time.strip()
        if len(time) == 4:  # "9:00" jaisa ho to "09:00"
            time = "0" + time
        if time not in SLOT_TIMES:
            return "Ye time valid nahi. Pehle check_availability se khali slot dekhein."

        try:
            day_bookings = get_active_bookings(d.isoformat())
            if any(r[2].strip() == phone_clean for r in day_bookings):
                return "Is number se is din pehle hi ek booking mojood hai."
            if time not in get_free_slots(d):
                return "Maazrat, ye slot abhi abhi bhar gaya. Koi aur time chunein."

            booking_id = f"SPC-{uuid.uuid4().hex[:6].upper()}"
            created = datetime.now(PKT).strftime("%Y-%m-%d %H:%M")
            get_worksheet().append_row(
                [booking_id, name, phone_clean, service_name, d.isoformat(), time, created, "Pending"],
                value_input_option="RAW",  # user ka text formula ban kar na chale
            )
        except Exception:
            logger.exception("Booking failed")
            return f"Booking system mein masla aaya. Clinic ko {CLINIC_PHONE} par call karein."

        session["bookings"] += 1
        session["booking_list"].append(
            f"{booking_id}: {name}, {service_name}, {d.strftime('%A, %d %B %Y')} at {slot_label(time)}"
        )
        note = ""
        if service_name == "Post-Surgery Rehabilitation":
            note = " Is service ke liye doctor ka referral saath layein."
        return (
            f"Booking save ho gayi. Booking ID: {booking_id}. {name}, {service_name}, "
            f"{d.strftime('%A, %d %B %Y')} ko {slot_label(time)}. Status: Pending. "
            f"Clinic call karke confirm karega.{note}"
        )

    return [search_clinic_info, check_availability, book_appointment]


# ---------------------------------------------------------------------------
# 6. System prompt (har request par aaj ki date ke saath)
# ---------------------------------------------------------------------------
def build_system_prompt(session: dict) -> str:
    now = datetime.now(PKT)
    if session["booking_list"]:
        booked = "\n".join(f"- {b}" for b in session["booking_list"])
    else:
        booked = "- None yet"
    next_days = "\n".join(
        f"- {(now + timedelta(days=i)).strftime('%A')}: {(now + timedelta(days=i)).strftime('%Y-%m-%d')}"
        for i in range(0, 8)
    )
    return f"""You are the assistant for Salman Physio Care, a physiotherapy clinic in Pakistan.

Current date and time (Pakistan): {now.strftime('%A, %Y-%m-%d, %I:%M %p')}
Upcoming dates, use these to convert words like aaj, kal, parson, or weekday names:
{next_days}

Language and style:
- Always reply in Roman Urdu or English, written in the Latin/English alphabet only. Never use Devanagari, Arabic, or any other script.
- Do not use markdown formatting. Never use ** or * or # symbols. Write in plain text only.
- For lists, start each item on a new line with a dash (-).
- Keep replies short and friendly.

Clinic information:
- For any question about services, fees, packages, timings, staff or policies, call search_clinic_info. Never invent fees or services.
- If the tool has no answer, say so honestly and suggest calling {CLINIC_PHONE}.

Booking an appointment:
1. Collect: full name, Pakistani mobile number, service, preferred date and time. Ask for missing details politely, one question per reply.
   - Never ask the user to type a date or time in any format (no YYYY-MM-DD, no 24-hour). Accept natural answers like "Monday", "kal", "4 baje", "shaam 5" and convert them yourself using the dates above.
   - Never repeat a question the user has already answered. Write only one short reply per turn.
2. Call check_availability for the date and offer the free slots.
3. Before booking, repeat all the details back and ask the user to confirm (for example: "Kya main ye booking kar doon?").
4. Only after the user clearly says yes, call book_appointment.
5. Share the Booking ID and tell them the clinic will call to confirm.
Never say a booking is done unless book_appointment returned a Booking ID.

Bookings already made in this chat (these are real and saved):
{booked}
- Never tell the user a booking was not made if it appears in this list.
- After a booking, if the user says "nahi", "no", "bas" or "shukriya", it means they need nothing else. Say goodbye politely. It does NOT cancel the booking.
- If the user wants to cancel or change a booking, ask them to call {CLINIC_PHONE} with their Booking ID.

Medical safety rules (always follow these first):
- Never diagnose any condition and never suggest or name medicines.
- If the user mentions severe or sudden pain, chest pain, numbness, weakness in arms or legs, loss of bladder or bowel control, a recent accident or fall, or a high fever, tell them to see a doctor or go to the emergency department immediately, before anything else.
- For specific health concerns, encourage an assessment with the physiotherapist.
"""


# ---------------------------------------------------------------------------
# 7. Sessions (har visitor ki chat history RAM mein)
# ---------------------------------------------------------------------------
sessions: dict[str, dict] = {}
SESSION_TIMEOUT = timedelta(hours=2)
MAX_SESSIONS = 300
MAX_HISTORY = 12  # LLM ko sirf aakhri 12 messages bheje jate hain (token bachat)


def get_or_create_session(session_id: str) -> dict:
    now = datetime.now(timezone.utc)
    expired = [sid for sid, s in sessions.items() if now - s["last_used"] > SESSION_TIMEOUT]
    for sid in expired:
        del sessions[sid]

    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            oldest = min(sessions, key=lambda sid: sessions[sid]["last_used"])
            del sessions[oldest]
        sessions[session_id] = {"history": [], "bookings": 0, "booking_list": [], "last_used": now}
    else:
        sessions[session_id]["last_used"] = now
    return sessions[session_id]


# ---------------------------------------------------------------------------
# 8. Rate limiting
# ---------------------------------------------------------------------------
def get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def global_key(request: Request) -> str:
    return "global"


limiter = Limiter(key_func=get_client_ip)

# ---------------------------------------------------------------------------
# 9. FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="Salman Physio Care Chatbot API")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Aap ne bohat zyada messages bhej diye hain. "
                      f"Thori der baad dobara try karein, ya clinic ko {CLINIC_PHONE} par call karein."
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


def extract_text(content) -> str:
    """LLM ka jawab kabhi string, kabhi list of parts hota hai."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


@app.api_route("/", methods=["GET", "HEAD"])
def health_check():
    return {"status": "ok", "service": "Salman Physio Care Chatbot"}


@app.post("/chat", response_model=ChatResponse)
@limiter.limit("15/minute")
@limiter.limit("100/day")
@limiter.limit("500/hour", key_func=global_key)
def chat(request: Request, req: ChatRequest):
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message khali nahi ho sakta.")

    session_id = req.session_id or str(uuid.uuid4())
    session = get_or_create_session(session_id)

    try:
        agent = create_agent(
            model=llm,
            tools=make_tools(session),
            system_prompt=build_system_prompt(session),
        )
        messages = session["history"][-MAX_HISTORY:] + [{"role": "user", "content": message}]
        result = agent.invoke({"messages": messages}, config={"recursion_limit": 12})
        answer = extract_text(result["messages"][-1].content).strip()
        if not answer:
            answer = f"Maazrat, jawab nahi ban saka. Clinic ko {CLINIC_PHONE} par call karein."
    except Exception:
        logger.exception("Agent failed for session %s", session_id)
        raise HTTPException(
            status_code=500,
            detail=f"Maazrat, abhi masla hai. Thori der baad try karein ya clinic ko {CLINIC_PHONE} par call karein.",
        )

    session["history"].extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ])
    return ChatResponse(answer=answer, session_id=session_id)


@app.post("/chat/reset")
@limiter.limit("10/minute")
def reset_session(request: Request, session_id: str):
    sessions.pop(session_id, None)
    return {"status": "reset", "session_id": session_id}
