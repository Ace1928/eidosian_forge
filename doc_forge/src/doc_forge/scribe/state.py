from __future__ import annotations

import fcntl
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Simple file lock wrapper for the whole write operation to ensure atomic replacement + exclusivity
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lockfile:
        fcntl.flock(lockfile, fcntl.LOCK_EX)
        try:
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp.replace(path)
        finally:
            fcntl.flock(lockfile, fcntl.LOCK_UN)


def read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        # Read with shared lock to prevent reading mid-write (though atomic replace mostly solves this)
        lock_path = path.with_suffix(path.suffix + ".lock")
        if lock_path.exists():
            with open(lock_path, "r") as lockfile:
                fcntl.flock(lockfile, fcntl.LOCK_SH)
                data = json.loads(path.read_text(encoding="utf-8"))
                fcntl.flock(lockfile, fcntl.LOCK_UN)
                return data
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


class ProcessorState:
    def __init__(self, state_path: Path, status_path: Path, index_path: Path):
        self.state_path = state_path
        self.status_path = status_path
        self.index_path = index_path
        self.lock = threading.Lock()
        self.data = self._load()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "started_at": now_iso(),
            "status": "initializing",
            "total_discovered": 0,
            "processed": 0,
            "approved": 0,
            "rejected": 0,
            "errors": 0,
            "average_seconds_per_document": 0.0,
            "average_quality_score": 0.0,
            "last_staged": None,
            "last_approved": None,
            "tag_frequency": {},
            "theme_frequency": {},
            "doc_type_frequency": {},
            "files": {},
            "index": [],
            "history": [],
        }

    def _load(self) -> Dict[str, Any]:
        state = read_json(self.state_path, self._default_state())
        default = self._default_state()
        for key, value in default.items():
            state.setdefault(key, value)
        return state

    def persist(self) -> None:
        with self.lock:
            snapshot = json.loads(json.dumps(self.data))
        atomic_write_json(self.state_path, snapshot)
        # Generate status payload
        total = int(snapshot.get("total_discovered", 0))
        processed = int(snapshot.get("processed", 0))
        remaining = max(0, total - processed)
        avg_seconds = float(snapshot.get("average_seconds_per_document", 0.0))
        eta = round(remaining * avg_seconds, 2) if avg_seconds > 0 else None

        status_payload = {
            "status": str(snapshot.get("status", "unknown")),
            "started_at": str(snapshot.get("started_at", now_iso())),
            "total_discovered": total,
            "processed": processed,
            "approved": int(snapshot.get("approved", 0)),
            "rejected": int(snapshot.get("rejected", 0)),
            "errors": int(snapshot.get("errors", 0)),
            "remaining": remaining,
            "average_seconds_per_document": float(snapshot.get("average_seconds_per_document", 0.0)),
            "average_quality_score": float(snapshot.get("average_quality_score", 0.0)),
            "eta_seconds": eta,
            "last_staged": snapshot.get("last_staged"),
            "last_approved": snapshot.get("last_approved"),
            "model": snapshot.get("model"),
            "completion_url": snapshot.get("completion_url"),
            "top_tags": sorted(snapshot.get("tag_frequency", {}).items(), key=lambda x: -x[1])[:12],
            "top_themes": sorted(snapshot.get("theme_frequency", {}).items(), key=lambda x: -x[1])[:12],
        }
        atomic_write_json(self.status_path, status_payload)
        atomic_write_json(self.index_path, {"entries": snapshot.get("index", [])})

    def update(self, key: str, value: Any) -> None:
        with self.lock:
            self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        with self.lock:
            return self.data.get(key, default)

    def update_running_average(self, key: str, value: float, count_key: str) -> None:
        with self.lock:
            current_count = int(self.data.get(count_key, 0))
            current_avg = float(self.data.get(key, 0.0))
            new_count = current_count + 1
            new_avg = ((current_avg * current_count) + value) / new_count
            self.data[key] = round(new_avg, 4)
            self.data[count_key] = new_count
