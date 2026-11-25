import os
import time
from dataclasses import dataclass

# Use LiteLLM's bundled local cost map so startup never blocks on GitHub fetch.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm.caching.caching import Cache
from litellm.exceptions import (
    APIConnectionError,
    BadGatewayError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)

from config import Settings

VALID_SIMULATED_STATUSES = {429, 500, 502, 503}


class AllProvidersUnavailableError(Exception):
    """Raised when every configured API key fails for a chat request."""


@dataclass
class LLMResult:
    reply: str
    resolution_source: str
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


def configure_litellm_cache(cache_enabled: bool) -> None:
    """Enable or disable LiteLLM's built-in in-memory response cache."""
    if cache_enabled:
        # In-memory cache keyed by model + messages (system prompt + history + user prompt).
        # API key and session_id are NOT part of LiteLLM's cache key, so identical
        # conversation states across different sessions still produce cache hits.
        litellm.cache = Cache(type="local")
    else:
        litellm.cache = None


def parse_failure_simulation(header_value: str | None) -> dict[str, int]:
    """
    Parse X-Simulate-Failure header values such as:
      - 'key_1=429'
      - 'key_1=429,key_2=500'
      - '*=429'
    """
    if not header_value:
        return {}

    simulations: dict[str, int] = {}
    for pair in header_value.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        target_key, raw_status = pair.split("=", 1)
        target_key = target_key.strip()
        try:
            status_code = int(raw_status.strip())
        except ValueError:
            continue
        if target_key and status_code in VALID_SIMULATED_STATUSES:
            simulations[target_key] = status_code
    return simulations


def _build_simulated_exception(status_code: int, key_id: str) -> Exception:
    """Create a realistic transient error for a simulated key failure."""
    msg = f"Simulated HTTP {status_code} failure for {key_id}"
    if status_code == 429:
        return RateLimitError(message=msg, llm_provider="simulated", model="simulated")
    if status_code == 502:
        return BadGatewayError(message=msg, llm_provider="simulated", model="simulated")
    if status_code == 503:
        return ServiceUnavailableError(
            message=msg, llm_provider="simulated", model="simulated"
        )
    return InternalServerError(message=msg, llm_provider="simulated", model="simulated")


def _is_retryable_error(exc: Exception) -> bool:
    """Return True if an exception represents rate limit, quota exhaustion, or 5xx."""
    if isinstance(
        exc,
        (
            RateLimitError,
            InternalServerError,
            ServiceUnavailableError,
            BadGatewayError,
            Timeout,
            APIConnectionError,
        ),
    ):
        return True

    status_code = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True

    lowered = str(exc).lower()
    return any(
        marker in lowered
        for marker in ("429", "rate limit", "quota", "resource_exhausted", "500", "502", "503")
    )


def generate_chat_reply(
    settings: Settings,
    messages: list[dict[str, str]],
    simulate_failure_header: str | None = None,
) -> LLMResult:
    """
    Call LiteLLM with automatic key failover and built-in response caching.

    - `messages` contains [system_message, ...history_messages, new_user_message].
    - `session_id` is intentionally excluded so LiteLLM can cache identical prompts
      across different sessions.
    - When `X-Simulate-Failure` is active, caching is bypassed for that request so
      failover behavior can be verified on demand.
    """
    if not settings.llm_api_keys:
        raise AllProvidersUnavailableError("all providers unavailable")

    simulations = (
        parse_failure_simulation(simulate_failure_header)
        if settings.enable_failure_simulation
        else {}
    )
    bypass_cache = bool(simulations)

    start_time = time.perf_counter()
    fallback_attempts = 0
    last_error: Exception | None = None

    for key_id, api_key in settings.llm_api_keys:
        # Check if this key (or '*' wildcard) is configured to fail for this request only.
        simulated_status = simulations.get(key_id) or simulations.get("*")
        if simulated_status is not None:
            last_error = _build_simulated_exception(simulated_status, key_id)
            fallback_attempts += 1
            continue

        call_kwargs: dict = {
            "model": settings.llm_model,
            "messages": messages,
            "api_key": api_key,
        }
        if settings.cache_enabled and bypass_cache:
            # Skip cache read/lookup while testing failover via X-Simulate-Failure.
            call_kwargs["cache"] = {"no-cache": True}

        try:
            response = litellm.completion(**call_kwargs)
        except Exception as exc:
            last_error = exc
            if _is_retryable_error(exc):
                fallback_attempts += 1
                continue
            # Non-retryable provider errors still count against this key and try next if available.
            fallback_attempts += 1
            continue

        latency_ms = max(0, int((time.perf_counter() - start_time) * 1000))

        # LiteLLM marks cache hits directly in response._hidden_params["cache_hit"].
        hidden_params = getattr(response, "_hidden_params", None) or {}
        is_cache_hit = bool(hidden_params.get("cache_hit")) and not bypass_cache

        reply_text = (response.choices[0].message.content or "").strip()
        if not reply_text:
            fallback_attempts += 1
            continue

        # Extract per-request token usage (prompt + completion for this single call only).
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return LLMResult(
            reply=reply_text,
            resolution_source="LLM_CACHE" if is_cache_hit else "LIVE_INFERENCE",
            model=settings.llm_model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            key_id=None if is_cache_hit else key_id,
            fallback_attempts=0 if is_cache_hit else fallback_attempts,
        )

    raise AllProvidersUnavailableError("all providers unavailable") from last_error
