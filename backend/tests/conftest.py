from types import SimpleNamespace
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure backend/ is importable when pytest runs from repo root or backend/.
BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import Settings  # noqa: E402
from main import create_app  # noqa: E402


def make_mock_completion_response(
    content: str = "Paris is the capital of France.",
    prompt_tokens: int = 25,
    completion_tokens: int = 8,
    cache_hit: bool = False,
):
    """Build a lightweight LiteLLM-compatible response object for unit tests."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
        _hidden_params={"cache_hit": cache_hit} if cache_hit else {},
    )


@pytest.fixture
def make_client(tmp_path):
    """Factory fixture creating an isolated TestClient with its own SQLite DB."""
    clients = []

    def _create(
        *,
        keys=None,
        enable_failure_simulation=True,
        cache_enabled=True,
    ) -> TestClient:
        db_file = tmp_path / f"test_{len(clients)}.db"
        settings = Settings(
            llm_api_keys=keys
            if keys is not None
            else [("key_1", "secret-1"), ("key_2", "secret-2")],
            llm_model="openai/gpt-4o-mini",
            database_url=f"sqlite:///{db_file}",
            sqlite_path=str(db_file),
            allowed_origins=["http://localhost:3000"],
            enable_failure_simulation=enable_failure_simulation,
            cache_enabled=cache_enabled,
        )
        app = create_app(custom_settings=settings)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client

    yield _create

    for client in clients:
        client.__exit__(None, None, None)
