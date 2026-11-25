import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load variables from a local .env file when running outside Docker.
load_dotenv()

DEFAULT_SYSTEM_PROMPT = (
    "You are LYFTR AI, a helpful, accurate, and concise AI assistant."
)


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_key_pool(raw_keys: str) -> list[tuple[str, str]]:
    """Convert comma-separated API keys into (key_id, secret_value) pairs."""
    keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
    return [(f"key_{index}", key) for index, key in enumerate(keys, start=1)]


def parse_sqlite_path(database_url: str) -> str:
    """Extract the filesystem path from a SQLite DATABASE_URL."""
    if database_url == ":memory:" or database_url == "sqlite:///:memory:":
        return ":memory:"
    if database_url.startswith("sqlite:///"):
        # Note: sqlite:////data/app.db keeps the leading '/' -> /data/app.db
        return database_url[len("sqlite:///") :]
    return database_url


@dataclass
class Settings:
    llm_api_keys: list[tuple[str, str]]
    llm_model: str
    database_url: str
    sqlite_path: str
    allowed_origins: list[str]
    enable_failure_simulation: bool
    cache_enabled: bool
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT


def load_settings() -> Settings:
    """Read application settings from environment variables."""
    raw_keys = os.getenv("LLM_API_KEYS", "")
    database_url = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")
    raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    allowed_origins = [
        origin.strip() for origin in raw_origins.split(",") if origin.strip()
    ]

    return Settings(
        llm_api_keys=parse_key_pool(raw_keys),
        llm_model=os.getenv("LLM_MODEL", "gemini/gemini-2.0-flash").strip(),
        database_url=database_url,
        sqlite_path=parse_sqlite_path(database_url),
        allowed_origins=allowed_origins,
        enable_failure_simulation=_parse_bool(
            os.getenv("ENABLE_FAILURE_SIMULATION"), default=False
        ),
        cache_enabled=_parse_bool(os.getenv("CACHE_ENABLED"), default=True),
    )
