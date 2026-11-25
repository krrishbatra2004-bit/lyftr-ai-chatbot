import json
import os
import sqlite3
from datetime import datetime, timezone


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Create the SQLite database file and messages table if they do not exist."""
    if db_path != ":memory:":
        parent_dir = os.path.dirname(db_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    with _connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_messages_session_id
            ON messages (session_id, id)
            """
        )


def check_db_health(db_path: str) -> bool:
    """Verify SQLite is reachable by running a lightweight query."""
    try:
        with _connect(db_path) as conn:
            conn.execute("SELECT 1").fetchone()
        return True
    except sqlite3.Error:
        return False


def get_session_messages(db_path: str, session_id: str) -> list[dict]:
    """Return all messages for a session ordered oldest-first."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, content, metadata
            FROM messages
            WHERE session_id = ?
            ORDER BY id ASC
            """,
            (session_id,),
        ).fetchall()

    messages: list[dict] = []
    for row in rows:
        item: dict = {
            "role": row["role"],
            "content": row["content"],
        }
        if row["metadata"] is not None:
            item["metadata"] = json.loads(row["metadata"])
        messages.append(item)
    return messages


def save_turn(
    db_path: str,
    session_id: str,
    user_message: str,
    assistant_reply: str,
    metadata: dict,
) -> None:
    """Persist one completed user/assistant turn in a single transaction."""
    now_iso = datetime.now(timezone.utc).isoformat()
    metadata_json = json.dumps(metadata)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, "user", user_message, None, now_iso),
        )
        conn.execute(
            """
            INSERT INTO messages (session_id, role, content, metadata, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, "assistant", assistant_reply, metadata_json, now_iso),
        )
