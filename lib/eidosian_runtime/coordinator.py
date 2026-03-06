from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


IDLE_PAYLOAD = {
    "contract": "eidos.runtime_coordinator.v1",
    "state": "idle",
    "task": "idle",
    "owner": "",
    "updated_at": "",
    "active_models": [],
    "metadata": {},
    "history": [],
}


def _forge_root() -> Path:
    raw = str(os.environ.get("EIDOS_FORGE_DIR") or os.environ.get("EIDOS_FORGE_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


class ForgeRuntimeCoordinator:
    def __init__(self, status_path: str | Path | None = None) -> None:
        root = _forge_root()
        self.status_path = (
            Path(status_path) if status_path else root / "data" / "runtime" / "forge_coordinator_status.json"
        )
        self.lock_path = self.status_path.with_suffix(self.status_path.suffix + ".lock")
        self._thread_lock = threading.RLock()
        self._lock_handle = None
        self._lock_depth = 0

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            if self._lock_depth == 0:
                self.lock_path.parent.mkdir(parents=True, exist_ok=True)
                self._lock_handle = open(self.lock_path, "a+", encoding="utf-8")
                if fcntl is not None:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
            self._lock_depth += 1
        try:
            yield
        finally:
            with self._thread_lock:
                self._lock_depth -= 1
                if self._lock_depth == 0 and self._lock_handle is not None:
                    if fcntl is not None:
                        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    self._lock_handle.close()
                    self._lock_handle = None

    def read(self) -> dict[str, Any]:
        with self._lock():
            return self._read_locked()

    def heartbeat(
        self,
        *,
        owner: str,
        task: str,
        state: str,
        active_models: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self._lock():
            payload = self._read_locked()
            history = list(payload.get("history") or [])
            history.append(
                {
                    "updated_at": _now_utc(),
                    "owner": str(owner),
                    "task": str(task),
                    "state": str(state),
                    "active_model_count": len(active_models or []),
                }
            )
            payload.update(
                {
                    "contract": IDLE_PAYLOAD["contract"],
                    "owner": str(owner),
                    "task": str(task),
                    "state": str(state),
                    "updated_at": _now_utc(),
                    "active_models": list(active_models or []),
                    "metadata": dict(metadata or {}),
                    "history": history[-24:],
                }
            )
            self._save_locked(payload)
            return payload

    def clear_owner(self, owner: str, *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock():
            payload = self._read_locked()
            if str(payload.get("owner") or "") == str(owner):
                extra = dict(payload.get("metadata") or {})
                extra.update(metadata or {})
                extra["released_owner"] = str(owner)
                payload["active_models"] = []
                payload["state"] = "idle"
                payload["task"] = "idle"
                payload["updated_at"] = _now_utc()
                payload["metadata"] = extra
                self._save_locked(payload)
            return payload

    def queue_snapshot(self, *, jobs: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock():
            payload = self._read_locked()
            meta = dict(payload.get("metadata") or {})
            meta["jobs"] = list(jobs)
            if policy is not None:
                meta["policy"] = dict(policy)
            payload["metadata"] = meta
            payload["updated_at"] = _now_utc()
            self._save_locked(payload)
            return payload

    def _read_locked(self) -> dict[str, Any]:
        if not self.status_path.exists():
            return dict(IDLE_PAYLOAD)
        try:
            payload = json.loads(self.status_path.read_text(encoding="utf-8"))
        except Exception:
            return dict(IDLE_PAYLOAD)
        if not isinstance(payload, dict):
            return dict(IDLE_PAYLOAD)
        payload.setdefault("contract", IDLE_PAYLOAD["contract"])
        payload.setdefault("state", "idle")
        payload.setdefault("task", "idle")
        payload.setdefault("owner", "")
        payload.setdefault("updated_at", "")
        payload.setdefault("active_models", [])
        payload.setdefault("metadata", {})
        payload.setdefault("history", [])
        return payload

    def _save_locked(self, payload: dict[str, Any]) -> None:
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.status_path.parent,
            prefix=f".{self.status_path.name}.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(json.dumps(payload, indent=2) + "\n")
        tmp_path.replace(self.status_path)
