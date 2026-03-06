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
        self.history_path = self.status_path.with_name("forge_runtime_trends.json")
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
            self._append_history_locked(payload)
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

    def history(self, limit: int = 120) -> list[dict[str, Any]]:
        with self._lock():
            payload = self._read_history_locked()
            entries = payload.get("entries") if isinstance(payload.get("entries"), list) else []
            return entries[-max(1, int(limit)) :]

    def can_allocate(
        self,
        *,
        owner: str,
        requested_models: list[dict[str, Any]] | None = None,
        allow_same_owner: bool = True,
    ) -> dict[str, Any]:
        requested_models = list(requested_models or [])
        with self._lock():
            payload = self._read_locked()
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            policy = metadata.get("policy") if isinstance(metadata.get("policy"), dict) else {}
            active_models = [row for row in (payload.get("active_models") or []) if isinstance(row, dict)]
            current_owner = str(payload.get("owner") or "")
            requested_families = {
                str(row.get("family") or "").strip().lower()
                for row in requested_models
                if str(row.get("family") or "").strip()
            }
            active_families = {
                str(row.get("family") or "").strip().lower()
                for row in active_models
                if str(row.get("family") or "").strip()
            }
            max_instances = max(1, int(policy.get("max_active_model_instances", 1) or 1))
            max_families = max(1, int(policy.get("max_active_model_families", 1) or 1))
            if allow_same_owner and current_owner and current_owner == str(owner):
                return {
                    "allowed": True,
                    "reason": "same_owner",
                    "active_model_count": len(active_models),
                    "max_active_model_instances": max_instances,
                    "max_active_model_families": max_families,
                }
            projected_instances = len(active_models) + len(requested_models)
            projected_families = (
                len(active_families | requested_families) if requested_families else len(active_families)
            )
            allowed = True
            reason = "ok"
            if projected_instances > max_instances:
                allowed = False
                reason = "instance_budget_exceeded"
            elif projected_families > max_families:
                allowed = False
                reason = "family_budget_exceeded"
            return {
                "allowed": allowed,
                "reason": reason,
                "active_model_count": len(active_models),
                "active_owner": current_owner,
                "max_active_model_instances": max_instances,
                "max_active_model_families": max_families,
            }

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

    def _read_history_locked(self) -> dict[str, Any]:
        if not self.history_path.exists():
            return {"contract": "eidos.runtime_trends.v1", "entries": []}
        try:
            payload = json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return {"contract": "eidos.runtime_trends.v1", "entries": []}
        if not isinstance(payload, dict):
            return {"contract": "eidos.runtime_trends.v1", "entries": []}
        payload.setdefault("contract", "eidos.runtime_trends.v1")
        payload.setdefault("entries", [])
        return payload

    def _append_history_locked(self, payload: dict[str, Any]) -> None:
        history = self._read_history_locked()
        entries = list(history.get("entries") or [])
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        entries.append(
            {
                "updated_at": payload.get("updated_at") or _now_utc(),
                "owner": str(payload.get("owner") or ""),
                "task": str(payload.get("task") or ""),
                "state": str(payload.get("state") or ""),
                "active_model_count": len(payload.get("active_models") or []),
                "active_models": list(payload.get("active_models") or []),
                "policy": dict(metadata.get("policy") or {}) if isinstance(metadata.get("policy"), dict) else {},
                "cycle": metadata.get("cycle"),
                "run_id": metadata.get("run_id"),
                "records_total": metadata.get("records_total"),
            }
        )
        history["entries"] = entries[-240:]
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self.history_path.parent,
            prefix=f".{self.history_path.name}.",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            handle.write(json.dumps(history, indent=2) + "\n")
        tmp_path.replace(self.history_path)
