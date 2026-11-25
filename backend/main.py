from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import Settings, load_settings
from database import check_db_health, get_session_messages, init_db, save_turn
from llm_service import (
    AllProvidersUnavailableError,
    configure_litellm_cache,
    generate_chat_reply,
)


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1)
    system_prompt: str | None = Field(default=None, max_length=4000)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed or len(trimmed) > 128:
            raise ValueError("session_id must be between 1 and 128 characters")
        return trimmed

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("message must not be empty")
        if len(trimmed) > 4000:
            raise ValueError("message must be at most 4000 characters after trimming")
        return trimmed


class ChatMetadata(BaseModel):
    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    metadata: ChatMetadata


class HealthResponse(BaseModel):
    status: str
    keys_configured: int
    cache_enabled: bool


def create_app(custom_settings: Settings | None = None) -> FastAPI:
    settings = custom_settings or load_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        init_db(settings.sqlite_path)
        configure_litellm_cache(settings.cache_enabled)
        yield

    app = FastAPI(title="LYFTR AI Backend", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins or ["http://localhost:3000"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_request: Request, _exc: Exception):
        # Always return structured JSON, never a stack trace or HTML page.
        return JSONResponse(
            status_code=500,
            content={"detail": "internal server error"},
        )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        current_settings: Settings = app.state.settings
        db_ok = check_db_health(current_settings.sqlite_path)
        if not db_ok:
            raise HTTPException(status_code=503, detail="database unavailable")

        return HealthResponse(
            status="ok",
            keys_configured=len(current_settings.llm_api_keys),
            cache_enabled=current_settings.cache_enabled,
        )

    @app.post("/chat", response_model=ChatResponse)
    def chat(
        payload: ChatRequest,
        x_simulate_failure: str | None = Header(default=None, alias="X-Simulate-Failure"),
    ) -> ChatResponse:
        current_settings: Settings = app.state.settings

        # Load prior turns for this session from SQLite (oldest first).
        history = get_session_messages(current_settings.sqlite_path, payload.session_id)

        system_content = (
            payload.system_prompt.strip()
            if payload.system_prompt and payload.system_prompt.strip()
            else current_settings.default_system_prompt
        )

        # Build the full prompt list: system prompt + prior turns + new user message.
        llm_messages = [{"role": "system", "content": system_content}]
        for item in history:
            llm_messages.append({"role": item["role"], "content": item["content"]})
        llm_messages.append({"role": "user", "content": payload.message})

        try:
            result = generate_chat_reply(
                settings=current_settings,
                messages=llm_messages,
                simulate_failure_header=x_simulate_failure,
            )
        except AllProvidersUnavailableError:
            raise HTTPException(status_code=503, detail="all providers unavailable")

        metadata = ChatMetadata(
            resolution_source=result.resolution_source,
            model=result.model,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            key_id=result.key_id,
            fallback_attempts=result.fallback_attempts,
        )

        # Persist user + assistant messages only after successful inference.
        save_turn(
            db_path=current_settings.sqlite_path,
            session_id=payload.session_id,
            user_message=payload.message,
            assistant_reply=result.reply,
            metadata=metadata.model_dump(),
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    @app.get("/sessions/{session_id}/messages")
    def get_messages(session_id: str) -> dict:
        current_settings: Settings = app.state.settings
        messages = get_session_messages(current_settings.sqlite_path, session_id)
        if not messages:
            raise HTTPException(status_code=404, detail="session not found")

        return {
            "session_id": session_id,
            "messages": messages,
        }

    return app


app = create_app()
