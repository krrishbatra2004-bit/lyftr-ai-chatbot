from unittest.mock import patch

from litellm.exceptions import InternalServerError, RateLimitError

from tests.conftest import make_mock_completion_response


def test_automatic_fallback_on_key_1_rate_limit(make_client):
    client = make_client()

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.side_effect = [
            RateLimitError(
                message="429 rate limit exceeded",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            ),
            make_mock_completion_response(content="Answered by backup key."),
        ]

        response = client.post(
            "/chat",
            json={"session_id": "s-fallback-1", "message": "Hello"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Answered by backup key."
    assert body["metadata"]["key_id"] == "key_2"
    assert body["metadata"]["fallback_attempts"] == 1
    assert body["metadata"]["resolution_source"] == "LIVE_INFERENCE"

    assert mock_completion.call_args_list[0].kwargs["api_key"] == "secret-1"
    assert mock_completion.call_args_list[1].kwargs["api_key"] == "secret-2"


def test_all_keys_failed_returns_503_and_does_not_persist(make_client):
    client = make_client()

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.side_effect = [
            RateLimitError(
                message="429 quota exceeded",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            ),
            InternalServerError(
                message="500 upstream error",
                llm_provider="openai",
                model="openai/gpt-4o-mini",
            ),
        ]

        response = client.post(
            "/chat",
            json={"session_id": "s-fail-all", "message": "Hello"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "all providers unavailable"}

    # Failed requests must never be stored in SQLite.
    history = client.get("/sessions/s-fail-all/messages")
    assert history.status_code == 404


def test_simulate_failure_header_failover_and_recovery(make_client):
    client = make_client(enable_failure_simulation=True)

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.return_value = make_mock_completion_response(
            content="Live answer from key 2"
        )
        res_429 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=429"},
            json={"session_id": "s-sim-1", "message": "Test failover"},
        )
        assert res_429.status_code == 200
        assert res_429.json()["metadata"]["key_id"] == "key_2"
        assert res_429.json()["metadata"]["fallback_attempts"] == 1
        assert mock_completion.call_args.kwargs["api_key"] == "secret-2"
        assert mock_completion.call_args.kwargs["cache"] == {"no-cache": True}

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.return_value = make_mock_completion_response(
            content="Live answer from key 2 on 500"
        )
        res_500 = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "key_1=500"},
            json={"session_id": "s-sim-2", "message": "Test 500 failover"},
        )
        assert res_500.status_code == 200
        assert res_500.json()["metadata"]["key_id"] == "key_2"
        assert res_500.json()["metadata"]["fallback_attempts"] == 1

    # Wildcard failure simulates all keys failing.
    res_all = client.post(
        "/chat",
        headers={"X-Simulate-Failure": "*=429"},
        json={"session_id": "s-sim-all", "message": "Should fail"},
    )
    assert res_all.status_code == 503
    assert res_all.json() == {"detail": "all providers unavailable"}
    assert client.get("/sessions/s-sim-all/messages").status_code == 404

    # Normal request immediately after simulation must still succeed on key_1.
    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.return_value = make_mock_completion_response(
            content="Key 1 recovered"
        )
        res_normal = client.post(
            "/chat",
            json={"session_id": "s-normal", "message": "Normal request"},
        )
    assert res_normal.status_code == 200
    assert res_normal.json()["metadata"]["key_id"] == "key_1"
    assert res_normal.json()["metadata"]["fallback_attempts"] == 0


def test_simulate_failure_ignored_when_disabled(make_client):
    client = make_client(enable_failure_simulation=False)

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.return_value = make_mock_completion_response(
            content="Simulation disabled"
        )
        response = client.post(
            "/chat",
            headers={"X-Simulate-Failure": "*=429"},
            json={"session_id": "s-disabled", "message": "Hello"},
        )

    assert response.status_code == 200
    assert response.json()["metadata"]["key_id"] == "key_1"
