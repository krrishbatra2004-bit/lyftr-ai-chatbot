from unittest.mock import patch

from tests.conftest import make_mock_completion_response


def test_unknown_session_returns_404(make_client):
    client = make_client()
    response = client.get("/sessions/does-not-exist/messages")
    assert response.status_code == 404
    assert response.json() == {"detail": "session not found"}


def test_session_history_order_and_metadata(make_client):
    client = make_client()

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.side_effect = [
            make_mock_completion_response(
                content="Hi! I am LYFTR AI.",
                prompt_tokens=20,
                completion_tokens=6,
            ),
            make_mock_completion_response(
                content="You asked me earlier to remember the number 42.",
                prompt_tokens=45,
                completion_tokens=11,
            ),
        ]

        r1 = client.post(
            "/chat",
            json={"session_id": "s-multi", "message": "Remember the number 42."},
        )
        r2 = client.post(
            "/chat",
            json={"session_id": "s-multi", "message": "What number did I say?"},
        )

    assert r1.status_code == 200
    assert r2.status_code == 200

    # Verify the second LLM call received the full conversation context.
    second_call_messages = mock_completion.call_args_list[1].kwargs["messages"]
    assert second_call_messages[0]["role"] == "system"
    assert second_call_messages[1:] == [
        {"role": "user", "content": "Remember the number 42."},
        {"role": "assistant", "content": "Hi! I am LYFTR AI."},
        {"role": "user", "content": "What number did I say?"},
    ]

    # Verify per-request token usage is NOT cumulative in the response.
    assert r1.json()["metadata"]["input_tokens"] == 20
    assert r2.json()["metadata"]["input_tokens"] == 45

    history = client.get("/sessions/s-multi/messages")
    assert history.status_code == 200
    data = history.json()
    assert data["session_id"] == "s-multi"
    assert len(data["messages"]) == 4

    # System prompt must never be returned, and oldest messages come first.
    assert [m["role"] for m in data["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert data["messages"][0]["content"] == "Remember the number 42."
    assert "metadata" not in data["messages"][0]
    assert data["messages"][1]["metadata"]["output_tokens"] == 6
    assert data["messages"][3]["metadata"]["input_tokens"] == 45


def test_sessions_stay_isolated(make_client):
    client = make_client()

    with patch("llm_service.litellm.completion") as mock_completion:
        mock_completion.side_effect = [
            make_mock_completion_response(content="Reply A"),
            make_mock_completion_response(content="Reply B"),
        ]
        client.post("/chat", json={"session_id": "sess-1", "message": "Hello A"})
        client.post("/chat", json={"session_id": "sess-2", "message": "Hello B"})

    # Second session should not have received sess-1's messages.
    sess2_messages = mock_completion.call_args_list[1].kwargs["messages"]
    assert len(sess2_messages) == 2
    assert sess2_messages[1]["content"] == "Hello B"

    h1 = client.get("/sessions/sess-1/messages").json()["messages"]
    h2 = client.get("/sessions/sess-2/messages").json()["messages"]
    assert len(h1) == 2
    assert len(h2) == 2
    assert h1[0]["content"] == "Hello A"
    assert h2[0]["content"] == "Hello B"
