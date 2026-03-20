from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_runtime_status(
    status_path: str | Path,
    *,
    contract: str,
    component: str,
    status: str,
    phase: str = "",
    message: str = "",
    history_path: str | Path | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload = {
        "contract": contract,
        "component": component,
        "status": status,
        "phase": phase,
        "message": message,
        "generated_at": _now_utc(),
    }
    payload.update(extra)
    path = Path(status_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if history_path is not None:
        history = Path(history_path)
        history.parent.mkdir(parents=True, exist_ok=True)
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    return payload
