# 🩺 Salman Physio Care — RAG Chatbot

An AI assistant for a physiotherapy clinic that answers patient questions about services, fees, packages, timings and booking, in **Roman Urdu and English**. It uses Retrieval-Augmented Generation (RAG) so every answer comes from the clinic's own data, with medical safety guardrails built in.

**🔗 Live demo:** https://salu12776.github.io/salman-physio-chatbot/

---

## ✨ Features

- **RAG pipeline**: answers are grounded in the clinic's documents, so the bot does not invent fees or services
- **Bilingual**: understands and replies in Roman Urdu and English, which is how most Pakistani patients actually type
- **Per-user conversation memory**: each visitor gets a separate session with summarised chat history
- **Medical safety guardrails**: never diagnoses or suggests medicines, and directs users with red-flag symptoms (e.g. severe pain after an accident, numbness, weakness) to a doctor or emergency department first
- **Production deployment**: static frontend on GitHub Pages, FastAPI backend on Render, kept warm with uptime monitoring

## 🏗️ Architecture

```mermaid
flowchart LR
    A[Visitor<br/>GitHub Pages site] -->|POST /chat| B[FastAPI backend<br/>Render]
    B -->|embed question| C[Cohere<br/>embeddings]
    B -->|similarity search| D[(Qdrant Cloud<br/>vector DB)]
    D -->|top-3 chunks| B
    B -->|context + question + history| E[Groq LLM<br/>gpt-oss-120b]
    E -->|answer| B
    B -->|JSON reply| A
```

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML, CSS, JavaScript (chat widget) on GitHub Pages |
| Backend | Python, FastAPI, Uvicorn |
| RAG framework | LangChain (ConversationalRetrievalChain, ConversationSummaryBufferMemory) |
| Embeddings | Cohere `embed-english-v3.0` |
| Vector database | Qdrant Cloud |
| LLM | `openai/gpt-oss-120b` via Groq |
| Hosting | Render (backend), GitHub Pages (frontend), UptimeRobot (monitoring) |

## 🔒 Security

- API keys are stored only as Render environment variables, never in the code or the repository
- CORS is restricted to the GitHub Pages domain, so other websites cannot call the backend from a browser
- Rate limiting with `slowapi`: per-user limits (15/minute, 100/day) plus a global limit (500/hour) to protect API credits
- Input validation: messages are capped at 500 characters
- Internal errors are logged on the server and never exposed to users
- In-memory sessions expire after 2 hours and are capped to prevent memory exhaustion

## 📁 Project Structure

```
├── index.html               # Clinic website with embedded chat widget
├── widget.html              # Standalone chat widget
├── main.py                  # FastAPI backend with RAG pipeline
├── physio_clinic_data.txt   # Clinic knowledge base (services, fees, policies)
├── requirements.txt         # Python dependencies
└── render.yaml              # Render deployment blueprint
```

## 🚀 Run Locally

```bash
git clone https://github.com/salu12776/salman-physio-chatbot.git
cd salman-physio-chatbot
pip install -r requirements.txt
```

Set these environment variables:

```
COHERE_API_KEY=your_cohere_key
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
QDRANT_COLLECTION=salman_physio_docs
ALLOWED_ORIGINS=http://localhost:5500
```

Start the server:

```bash
uvicorn main:app --reload
```

## 🔌 API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/chat` | Send `{"message": "...", "session_id": "..."}`, returns `{"answer": "...", "session_id": "..."}` |
| POST | `/chat/reset?session_id=...` | Clear a session's memory |

## 🗺️ Roadmap

- [ ] Appointment booking through tool calling (agentic workflow)
- [ ] WhatsApp integration
- [ ] Persistent session memory with Redis
- [ ] Multilingual embeddings for better Roman Urdu retrieval

## 👤 Author

**Salman** — [GitHub](https://github.com/salu12776)
https://docs.google.com/spreadsheets/d/1dek9Rr9kZs3F-ly5oakLij8ZjkTsumD-qZ5Cwz90RgE/edit?gid=0#gid=0
