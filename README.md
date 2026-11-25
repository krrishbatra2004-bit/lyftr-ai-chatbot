# LYFTR AI — Resilient LLM Chatbot

A full-stack AI chatbot built with React (Vite), FastAPI, LiteLLM, and SQLite featuring multi-turn session memory, automatic API key failover, built-in response caching, and per-request token tracking.

## 1. Live URLs

- **Frontend (Vercel):** Not yet deployed (configure `VITE_API_BASE_URL` to point to your deployed backend URL)
- **Backend (Render):** Not yet deployed (deploy using `render.yaml` with a persistent disk at `/data`)
- **Health Check:** `<BACKEND_URL>/health`

> Replace the placeholders above once you deploy the backend to Render and the frontend to Vercel.

---

## 2. Run locally: with Docker and without

### Prerequisites

- Copy `.env.example` to `.env` and add at least two provider API keys:

```bash
cp .env.example .env
```

Example `.env`:

```env
LLM_API_KEYS=your_first_api_key,your_second_api_key
LLM_MODEL=gemini/gemini-2.0-flash
DATABASE_URL=sqlite:////data/app.db
ALLOWED_ORIGINS=http://localhost:3000
ENABLE_FAILURE_SIMULATION=true
CACHE_ENABLED=true
VITE_API_BASE_URL=http://localhost:8000
```

### Option A: With Docker (Recommended)

Start both containers using the `Makefile` (or `docker-compose` directly):

```bash
make up
```

Services will be available at:
- **Frontend:** http://localhost:3000
- **Backend:** http://localhost:8000
- **Health check:** http://localhost:8000/health

Stop both services (SQLite data persists in the named Docker volume `lyftr_data` mounted at `/data`):

```bash
make down
```

### Option B: Without Docker

**1. Start the Backend**

```bash
cd backend
python -m venv .venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

*(When running outside Docker without `DATABASE_URL` set to `/data/app.db`, the backend defaults to `./data/app.db`.)*

**2. Start the Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000 in your browser.

**3. Run Backend Tests**

```bash
cd backend
pytest -v
```

---

## 3. Design decisions: key fallback, caching, SQLite session storage

### Key Fallback (`backend/llm_service.py`)
- `LLM_API_KEYS` is parsed into an ordered key pool (`key_1`, `key_2`, ...).
- On each `POST /chat` request, the backend tries keys sequentially using `litellm.completion(..., api_key=...)`.
- If a key raises a rate-limit (`429`), quota exhaustion, timeout, or `5xx` error, the service increments `fallback_attempts` and automatically tries the next key without exposing the secret key or crashing the request.
- If all keys fail, the API returns HTTP `503` with `{"detail": "all providers unavailable"}` and saves nothing to SQLite.

### Response Caching (`backend/llm_service.py`)
- Uses LiteLLM's built-in in-memory cache: `litellm.cache = Cache(type="local")`.
- The cache key is computed by LiteLLM from the `model` and the full `messages` array (`system_prompt + conversation_history + new_message`).
- Neither `session_id` nor `api_key` is included in the cache key, so the same prompt across two different sessions hits the cache (`LLM_CACHE`, `key_id: null`, `fallback_attempts: 0`), while a different conversation history produces a fresh `LIVE_INFERENCE` response.
- Cache hits are detected directly via `response._hidden_params.get("cache_hit")` and preserve the original cached `input_tokens` and `output_tokens`.
- Requests with `X-Simulate-Failure` pass `cache={"no-cache": True}` so failover testing always exercises the key pool instead of returning a cached answer.

### SQLite Session Storage (`backend/database.py`)
- Uses Python's standard `sqlite3` module directly with the `messages` table (`id`, `session_id`, `role`, `content`, `metadata`, `created_at`) created automatically on startup.
- Before calling LiteLLM, `POST /chat` loads previous turns for `session_id` ordered oldest-first so the model has full multi-turn context.
- Both the user message and assistant response (with serialized JSON metadata) are saved in a single transaction **only after** inference succeeds, ensuring failed `422` or `503` requests never enter session history.
- `GET /sessions/{session_id}/messages` returns stored history oldest-first without the system prompt (or `404` if the session has no messages).

---

## 4. Verification: how to see failover, cache hits and per-message token usage

### 1. Check Health

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "keys_configured": 2,
  "cache_enabled": true
}
```

### 2. Verify Cache Hit Across Two Sessions

Send the first request in `session-a`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-a", "message": "What is the capital of France?"}'
```

Expected metadata: `"resolution_source": "LIVE_INFERENCE"`, `"key_id": "key_1"`.

Send the identical prompt in a brand-new `session-b`:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "session-b", "message": "What is the capital of France?"}'
```

Expected metadata: `"resolution_source": "LLM_CACHE"`, `"key_id": null`, `"fallback_attempts": 0`, identical `reply`, same `input_tokens`/`output_tokens`, and lower `latency_ms`.

### 3. Verify Key Failover with `X-Simulate-Failure`

Simulate `key_1` returning HTTP `429` (requires `ENABLE_FAILURE_SIMULATION=true`):

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: key_1=429" \
  -d '{"session_id": "session-failover", "message": "Explain HTTP 429 in one sentence."}'
```

Expected metadata: HTTP `200`, `"key_id": "key_2"`, `"fallback_attempts": 1`, `"resolution_source": "LIVE_INFERENCE"`.

Simulate all keys failing (`*=429`):

```bash
curl -i -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Simulate-Failure: *=429" \
  -d '{"session_id": "session-down", "message": "Hello"}'
```

Expected response: HTTP `503` with `{"detail": "all providers unavailable"}`.

### 4. Verify Multi-Turn Memory and Per-Message Token Usage

Send a follow-up turn in the same session and inspect stored history:

```bash
curl http://localhost:8000/sessions/session-failover/messages
```

Each assistant message includes its own per-request `input_tokens` and `output_tokens` (prompt tokens grow naturally with conversation history, rather than accumulating a running total). The UI also displays `resolution_source`, `model`, `latency_ms`, `input_tokens`, and `output_tokens` beneath every assistant reply.

---

## 5. Setup Used: your editor and AI tools

- **Code Editor:** VS Code / Antigravity IDE
- **AI Coding Assistant:** Antigravity AI coding assistant
- **Testing & CLI:** Pytest, HTTPX, Vite, Docker Compose, curl

---

## 6. Bonus features (if any): what you added beyond the requirements and why

- **Live Backend Connection & Key/Cache Status Indicator:** The sidebar calls `GET /health` on startup so users can immediately confirm the backend is reachable, how many keys are configured, and whether caching is enabled.
- **Suggested Starter Prompts:** Quick-click prompt chips on the welcome screen make it effortless to test caching and multi-turn chat during demos.
- **Render Blueprint (`render.yaml`) & Vercel SPA Config (`frontend/vercel.json`):** Added ready-to-use deployment manifests with persistent `/data` disk configuration so deployment takes minimal manual setup.
