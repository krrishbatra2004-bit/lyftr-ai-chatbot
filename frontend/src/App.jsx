import { useEffect, useRef, useState } from "react";

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000"
).replace(/\/+$/, "");

const SUGGESTED_QUESTIONS = [
  "What is the capital of France?",
  "Explain cloud computing in simple terms.",
  "Why does API key failover matter in AI apps?",
];

function createSessionId() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return `s-${crypto.randomUUID().slice(0, 8)}`;
  }
  return `s-${Math.random().toString(36).slice(2, 10)}`;
}

function formatErrorDetail(detail, status) {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((item) => item.msg || JSON.stringify(item)).join("; ");
  }
  return `Request failed with status ${status}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(() => createSessionId());
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [pendingPrompt, setPendingPrompt] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState({
    status: "checking",
    keys_configured: 0,
    cache_enabled: false,
  });

  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    let active = true;

    async function checkHealth() {
      try {
        const res = await fetch(`${API_BASE_URL}/health`);
        if (!res.ok) {
          throw new Error(`HTTP ${res.status}`);
        }
        const data = await res.json();
        if (active) {
          setHealth({
            status: data.status === "ok" ? "ok" : "error",
            keys_configured: data.keys_configured ?? 0,
            cache_enabled: Boolean(data.cache_enabled),
          });
        }
      } catch {
        if (active) {
          setHealth({
            status: "error",
            keys_configured: 0,
            cache_enabled: false,
          });
        }
      }
    }

    checkHealth();
    const retryTimer = setInterval(() => {
      if (active) {
        checkHealth();
      }
    }, 5000);

    return () => {
      active = false;
      clearInterval(retryTimer);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, pendingPrompt, isLoading, error]);

  async function sendMessage(overrideText) {
    if (isLoading) return;

    const rawText = typeof overrideText === "string" ? overrideText : input;
    const trimmed = rawText.trim();

    setError(null);

    if (!trimmed && rawText.length === 0) {
      setError("Please enter a message before sending.");
      return;
    }

    setInput("");
    setPendingPrompt(trimmed || rawText);
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: rawText,
        }),
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const detail = formatErrorDetail(data?.detail, response.status);
        setError(`Error (${response.status}): ${detail}`);
        return;
      }

      const now = Date.now();
      setMessages((prev) => [
        ...prev,
        {
          id: `user-${now}`,
          role: "user",
          content: trimmed,
        },
        {
          id: `assistant-${now}`,
          role: "assistant",
          content: data.reply,
          metadata: data.metadata,
        },
      ]);
    } catch {
      setError(
        "Unable to reach the backend service. Please verify the API server is running."
      );
    } finally {
      setPendingPrompt(null);
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }

  function handleSubmit(event) {
    event.preventDefault();
    sendMessage();
  }

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (input.trim() && !isLoading) {
        sendMessage();
      }
    }
  }

  function handleNewChat() {
    setSessionId(createSessionId());
    setMessages([]);
    setError(null);
    setInput("");
    setPendingPrompt(null);
    inputRef.current?.focus();
  }

  const isEmpty = messages.length === 0 && !pendingPrompt;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand">
            <div className="brand-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z" />
                <path d="M12 12l8-4.5M12 12v9M12 12L4 7.5" />
              </svg>
            </div>
            <div>
              <h1 className="brand-name">LYFTR AI</h1>
              <p className="brand-subtitle">Resilient LLM Gateway</p>
            </div>
          </div>

          <button
            type="button"
            className="new-chat-btn"
            onClick={handleNewChat}
            disabled={isLoading}
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 5v14M5 12h14" />
            </svg>
            New Chat
          </button>

          <div className="session-card">
            <div className="session-label">Current Session</div>
            <div className="session-id" title={sessionId}>
              {sessionId}
            </div>
            <div className="session-stats">
              {messages.length} {messages.length === 1 ? "message" : "messages"} stored
            </div>
          </div>
        </div>

        <div className="sidebar-footer">
          <div className="status-row">
            <span
              className={`status-dot ${
                health.status === "ok"
                  ? "online"
                  : health.status === "checking"
                  ? "checking"
                  : "offline"
              }`}
            />
            <span className="status-text">
              {health.status === "ok"
                ? "Backend connected"
                : health.status === "checking"
                ? "Checking backend..."
                : "Backend unreachable"}
            </span>
          </div>
          {health.status === "ok" && (
            <div className="status-details">
              {health.keys_configured} API {health.keys_configured === 1 ? "key" : "keys"} · Cache{" "}
              {health.cache_enabled ? "ON" : "OFF"}
            </div>
          )}
        </div>
      </aside>

      <main className="chat-main">
        <div className="chat-scroll">
          {isEmpty ? (
            <section className="welcome-panel">
              <div className="welcome-badge">Multi-turn · Key Failover · Response Cache</div>
              <h2 className="welcome-title">Welcome to LYFTR AI</h2>
              <p className="welcome-description">
                Ask anything below. Conversations are stored per session in SQLite,
                responses are cached with LiteLLM, and requests fail over automatically
                between backup API keys.
              </p>

              <div className="suggestions">
                {SUGGESTED_QUESTIONS.map((question) => (
                  <button
                    key={question}
                    type="button"
                    className="suggestion-chip"
                    onClick={() => sendMessage(question)}
                    disabled={isLoading}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </section>
          ) : (
            <div className="messages-list">
              {messages.map((msg) =>
                msg.role === "user" ? (
                  <article key={msg.id} className="message-row user-row">
                    <div className="message-bubble user-bubble">{msg.content}</div>
                  </article>
                ) : (
                  <article
                    key={msg.id}
                    className="message-row assistant-row"
                    data-testid="assistant-message"
                  >
                    <div className="assistant-avatar" aria-hidden="true">
                      AI
                    </div>
                    <div className="assistant-body">
                      <div className="message-bubble assistant-bubble">{msg.content}</div>
                      {msg.metadata && (
                        <div className="message-meta">
                          <span
                            className={`meta-badge ${
                              msg.metadata.resolution_source === "LLM_CACHE"
                                ? "cache"
                                : "live"
                            }`}
                            data-testid="meta-source"
                          >
                            {msg.metadata.resolution_source}
                          </span>
                          <span className="meta-item">
                            <span className="meta-label">Model:</span>
                            <span data-testid="meta-model">{msg.metadata.model}</span>
                          </span>
                          <span className="meta-item">
                            <span className="meta-label">Latency:</span>
                            <span data-testid="meta-latency">
                              {msg.metadata.latency_ms} ms
                            </span>
                          </span>
                          <span className="meta-item">
                            <span className="meta-label">In:</span>
                            <span data-testid="meta-input-tokens">
                              {msg.metadata.input_tokens}
                            </span>
                          </span>
                          <span className="meta-item">
                            <span className="meta-label">Out:</span>
                            <span data-testid="meta-output-tokens">
                              {msg.metadata.output_tokens}
                            </span>
                          </span>
                          {msg.metadata.key_id && (
                            <span className="meta-item">
                              <span className="meta-label">Key:</span>
                              <span>
                                {msg.metadata.key_id}
                                {msg.metadata.fallback_attempts > 0
                                  ? ` (${msg.metadata.fallback_attempts} fallback)`
                                  : ""}
                              </span>
                            </span>
                          )}
                        </div>
                      )}
                    </div>
                  </article>
                )
              )}

              {pendingPrompt && (
                <article className="message-row user-row">
                  <div className="message-bubble user-bubble">{pendingPrompt}</div>
                </article>
              )}

              {isLoading && (
                <div className="message-row assistant-row" aria-live="polite">
                  <div className="assistant-avatar" aria-hidden="true">
                    AI
                  </div>
                  <div className="loading-bubble">
                    <span className="dot" />
                    <span className="dot" />
                    <span className="dot" />
                    <span className="loading-text">Thinking...</span>
                  </div>
                </div>
              )}
            </div>
          )}

          {error && (
            <div className="error-banner" role="alert">
              <strong>Could not complete request:</strong> {error}
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="composer-bar" onSubmit={handleSubmit}>
          <div className="composer-box">
            <textarea
              ref={inputRef}
              data-testid="chat-input"
              className="chat-input"
              rows={1}
              placeholder="Ask LYFTR AI a question... (Enter to send, Shift+Enter for new line)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              type="submit"
              data-testid="send-button"
              className="send-button"
              disabled={isLoading}
              aria-label="Send message"
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 2L11 13" />
                <path d="M22 2L15 22l-4-9-9-4 20-7z" />
              </svg>
              <span>Send</span>
            </button>
          </div>
        </form>
      </main>
    </div>
  );
}
