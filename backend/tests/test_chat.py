from unittest.mock import patch

import litellm

from tests.conftest import make_mock_completion_response


def test_health_endpoint(make_client):
    client = make_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "keys_configured": 2,
        "cache_enabled": True,
    }


def test_valid_chat_request_returns_expected_contract(make_client):
    client = make_client()

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.return_value = make_mock_completion_response(
            content="The capital of France is Paris.",
            prompt_tokens=42,
            completion_tokens=9,
        )

        response = client.post(
            "/chat",
            json={
                "session_id": "s-valid-1",
                "message": "What is the capital of France?",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == "s-valid-1"
    assert body["reply"] == "The capital of France is Paris."
    assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert body["metadata"]["model"] == "openai/gpt-4o-mini"
    assert isinstance(body["metadata"]["latency_ms"], int)
    assert body["metadata"]["input_tokens"] == 42
    assert body["metadata"]["output_tokens"] == 9
    assert body["metadata"]["key_id"] == "key_1"
    assert body["metadata"]["fallback_attempts"] == 0


def test_empty_and_whitespace_messages_return_422(make_client):
    client = make_client()

    for bad_message in ["", "   ", "\n\t  "]:
        response = client.post(
            "/chat",
            json={"session_id": "s-empty", "message": bad_message},
        )
        assert response.status_code == 422
        assert "detail" in response.json()

    # Failed validation requests must not create session history.
    history_res = client.get("/sessions/s-empty/messages")
    assert history_res.status_code == 404


def test_message_length_validation(make_client):
    client = make_client()

    too_long = "a" * 4001
    res_long = client.post(
        "/chat",
        json={"session_id": "s-long", "message": too_long},
    )
    assert res_long.status_code == 422
    assert "detail" in res_long.json()

    # Exactly 4000 characters after trimming outer whitespace must succeed.
    trimmed_4000 = "  " + ("a" * 4000) + "  "
    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.return_value = make_mock_completion_response(content="OK")
        res_ok = client.post(
            "/chat",
            json={"session_id": "s-long-ok", "message": trimmed_4000},
        )
    assert res_ok.status_code == 200


def test_real_litellm_cache_hit_and_miss_across_sessions(make_client):
    client = make_client(cache_enabled=True)

    original_completion = litellm.completion
    live_calls = 0

    def fake_provider_completion(**kwargs):
        # Delegate to real litellm.completion with mock_response so LiteLLM's
        # built-in Cache(type="local") executes end-to-end without network calls.
        nonlocal live_calls
        if not kwargs.get("cache", {}).get("no-cache"):
            # Count invocations when LiteLLM actually calls us; note mock_response
            # still passes through LiteLLM's cache lookup first.
            pass
        user_msg = kwargs["messages"][-1]["content"]
        if "France" in user_msg:
            mock_text = "The capital of France is Paris."
        else:
            mock_text = "The capital of Japan is Tokyo."
        response = original_completion(mock_response=mock_text, **kwargs)
        if not (response._hidden_params or {}).get("cache_hit"):
            live_calls += 1
        return response

    with patch("llm_service.litellm.completion", side_effect=fake_provider_completion):
        first = client.post(
            "/chat",
            json={"session_id": "session-a", "message": "What is the capital of France?"},
        )
        second = client.post(
            "/chat",
            json={"session_id": "session-b", "message": "What is the capital of France?"},
        )
        different = client.post(
            "/chat",
            json={"session_id": "session-c", "message": "What is the capital of Japan?"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert different.status_code == 200

    first_data = first.json()
    second_data = second.json()
    different_data = different.json()

    assert first_data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert first_data["metadata"]["key_id"] == "key_1"

    assert second_data["metadata"]["resolution_source"] == "LLM_CACHE"
    assert second_data["metadata"]["key_id"] is None
    assert second_data["metadata"]["fallback_attempts"] == 0
    assert second_data["reply"] == first_data["reply"]
    assert second_data["metadata"]["input_tokens"] == first_data["metadata"]["input_tokens"]
    assert second_data["metadata"]["output_tokens"] == first_data["metadata"]["output_tokens"]

    assert different_data["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert different_data["reply"] != first_data["reply"]
    assert live_calls == 2


def test_different_conversation_history_does_not_hit_cache(make_client):
    client = make_client(cache_enabled=True)
    original_completion = litellm.completion

    def fake_provider_completion(**kwargs):
        messages = kwargs["messages"]
        if len(messages) == 2:
            mock_text = "Standalone answer: Paris."
        else:
            mock_text = "Follow-up answer with history: Paris."
        return original_completion(mock_response=mock_text, **kwargs)

    with patch("llm_service.litellm.completion", side_effect=fake_provider_completion):
        standalone = client.post(
            "/chat",
            json={"session_id": "s-hist-a", "message": "What is the capital of France?"},
        )
        client.post(
            "/chat",
            json={"session_id": "s-hist-b", "message": "We are talking about Europe."},
        )
        follow_up = client.post(
            "/chat",
            json={"session_id": "s-hist-b", "message": "What is the capital of France?"},
        )

    assert standalone.status_code == 200
    assert follow_up.status_code == 200
    assert standalone.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert follow_up.json()["metadata"]["resolution_source"] == "LIVE_INFERENCE"
    assert follow_up.json()["reply"] != standalone.json()["reply"]

