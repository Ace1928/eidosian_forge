import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
from eidosian_core import eidosian


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


@eidosian()
def log_event(event: Dict[str, Any]) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        _rotate_if_needed()
        event["timestamp"] = _utc_now()
        with LOG_FILE.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=True))
            handle.write("\n")
    except Exception:
        # Fallback: don't crash the server if logging fails
        pass


@eidosian()
def log_tool_call(name: str, arguments: Dict[str, Any], result: Any, error: str | None = None, start_time: float | None = None) -> None:
    duration = None
    if start_time:
        duration = time.time() - start_time
    
    event = {
        "type": "tool",
        "name": name,
        "arguments": arguments,
        "result": result,
        "error": error,
        "duration": duration,
    }
    if error:
        event["traceback"] = traceback.format_exc()
        
    log_event(event)


@eidosian()
def log_resource_read(uri: str, result: Any, error: str | None = None, start_time: float | None = None) -> None:
    duration = None
    if start_time:
        duration = time.time() - start_time

    event = {
        "type": "resource",
        "uri": uri,
        "result": result,
        "error": error,
        "duration": duration,
    }
    if error:
        event["traceback"] = traceback.format_exc()

    log_event(event)


@eidosian()
def log_startup(transport: str) -> None:
    log_event(
        {
            "type": "startup",
            "transport": transport,
            "message": "Eidosian MCP Server started",
        }
    )


@eidosian()
def log_error(context: str, error: str) -> None:
    log_event(
        {
            "type": "error",
            "context": context,
            "error": error,
        }
    )
