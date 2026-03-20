from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from eidosian_runtime.session_bridge import (
    build_session_context,
    import_codex_rollouts,
    import_gemini_journal,
    read_session_events,
    summarize_import_status,
)


def test_import_gemini_journal_deduplicates(tmp_path: Path) -> None:
    home = tmp_path / "home"
    journal = home / ".gemini" / "context_memory" / "user.journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        "\n".join(
            [
                json.dumps({"id": "a1", "text": "first gemini memory", "scope": "global", "op": "upsert"}),
                json.dumps({"id": "a2", "text": "second gemini memory", "scope": "global", "op": "upsert"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    events_path = tmp_path / "events.jsonl"
    status_path = tmp_path / "import_status.json"
    first = import_gemini_journal(home_root=home, events_path=events_path, import_status_path=status_path, limit=10)
    second = import_gemini_journal(home_root=home, events_path=events_path, import_status_path=status_path, limit=10)
    assert first["imported"] == 2
    assert second["imported"] == 0
    rows = read_session_events(limit=10, events_path=events_path)
    assert [row["summary"] for row in rows] == ["first gemini memory", "second gemini memory"]


def test_import_codex_rollouts_reads_recent_threads(tmp_path: Path) -> None:
    home = tmp_path / "home"
    codex = home / ".codex"
    codex.mkdir(parents=True, exist_ok=True)
    rollout = codex / "sessions" / "2026" / "03" / "20" / "rollout.jsonl"
    rollout.parent.mkdir(parents=True, exist_ok=True)
    rollout.write_text(
        "\n".join(
            [
                json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hello from codex"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "input_text", "text": "response from codex"}],
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    db = sqlite3.connect(str(codex / "state_5.sqlite"))
    db.execute(
        "create table threads (id text primary key, rollout_path text not null, created_at integer not null, updated_at integer not null, source text not null, model_provider text not null, cwd text not null, title text not null)"
    )
    db.execute(
        "insert into threads (id, rollout_path, created_at, updated_at, source, model_provider, cwd, title) values (?, ?, ?, ?, ?, ?, ?, ?)",
        ("thread-1", str(rollout), 1, 10, "cli", "openai", str(home), "test"),
    )
    db.commit()
    db.close()

    events_path = tmp_path / "events.jsonl"
    status_path = tmp_path / "import_status.json"
    result = import_codex_rollouts(
        home_root=home,
        events_path=events_path,
        import_status_path=status_path,
        thread_limit=2,
        events_per_thread=4,
    )
    assert result["imported"] == 2
    rows = read_session_events(limit=10, events_path=events_path)
    assert rows[0]["interface"] == "codex"
    assert rows[0]["event_type"] == "user_import"
    assert rows[1]["event_type"] == "assistant_import"


def test_build_session_context_syncs_external_sessions(tmp_path: Path) -> None:
    home = tmp_path / "home"
    journal = home / ".gemini" / "context_memory" / "user.journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(json.dumps({"id": "g1", "text": "gemini continuity", "scope": "global", "op": "upsert"}) + "\n", encoding="utf-8")

    from eidosian_runtime import session_bridge as mod

    mod.HOME_ROOT = home
    mod.DEFAULT_EVENTS_PATH = tmp_path / "events.jsonl"
    mod.DEFAULT_STATUS_PATH = tmp_path / "latest_context.json"
    mod.DEFAULT_IMPORT_STATUS_PATH = tmp_path / "import_status.json"
    payload = build_session_context(interface="qwenchat", query="continuity", session_id="qwenchat:test")
    assert payload["recent_sessions"]
    assert (tmp_path / "latest_context.json").exists()


def test_summarize_import_status_treats_codex_thread_values_as_versions_not_counts() -> None:
    payload = summarize_import_status(
        {
            "last_sync_at": "2026-03-20T00:00:00Z",
            "gemini": {"imported_ids": ["g1", "g2"]},
            "codex": {
                "threads": {
                    "thread-a": 1773975905,
                    "thread-b": 1772214073,
                }
            },
        }
    )
    assert payload["gemini_records"] == 2
    assert payload["codex_records"] == 2
    assert payload["codex_thread_count"] == 2
    assert payload["imported_records"] == 4
