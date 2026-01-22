from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


LOG_DIR = Path("~/.eidosian/mcp_logs").expanduser()
LOG_FILE = LOG_DIR / "messages.jsonl"
MAX_BYTES = 10 * 1024 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rotate_if_needed() -> None:
    if LOG_FILE.exists() and LOG_FILE.stat().st_size > MAX_BYTES:
        rotated = LOG_DIR / "messages.jsonl.1"
        if rotated.exists():
            rotated.unlink()
        LOG_FILE.rename(rotated)


def log_event(event: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_if_needed()
    event["timestamp"] = _utc_now()
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=True))
        handle.write("\n")


def log_tool_call(name: str, arguments: Dict[str, Any], result: Any, error: str | None = None) -> None:
    log_event(
        {
            "type": "tool",
            "name": name,
            "arguments": arguments,
            "result": result,
            "error": error,
        }
    )


def log_resource_read(uri: str, result: Any, error: str | None = None) -> None:
    log_event(
        {
            "type": "resource",
            "uri": uri,
            "result": result,
            "error": error,
        }
    )
