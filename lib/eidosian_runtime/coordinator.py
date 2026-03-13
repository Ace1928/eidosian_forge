from __future__ import annotations

import json
import os
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
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
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_utc(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _pid_alive(value: Any) -> bool:
    try:
        pid = int(value)
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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

    def trend_summary(self, limit: int = 120) -> dict[str, Any]:
        entries = self.history(limit=max(1, int(limit)))
        active_model_counts: list[int] = []
        docs_coverage_values: list[float] = []
        docs_missing_values: list[int] = []
        state_counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        saturated = 0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                active_model_counts.append(max(0, int(entry.get("active_model_count") or 0)))
            except Exception:
                active_model_counts.append(0)
            state = str(entry.get("state") or "").strip().lower()
            if state:
                state_counts[state] = state_counts.get(state, 0) + 1
            task = str(entry.get("task") or "").strip().lower()
            if task:
                task_counts[task] = task_counts.get(task, 0) + 1
            metadata = entry.get("metadata") if isinstance(entry.get("metadata"), dict) else {}
            try:
                docs_coverage_values.append(float(metadata.get("directory_docs_coverage") or 0.0))
            except Exception:
                pass
            try:
                docs_missing_values.append(max(0, int(metadata.get("directory_docs_missing") or 0)))
            except Exception:
                pass
            policy = entry.get("policy") if isinstance(entry.get("policy"), dict) else {}
            try:
                max_instances = max(1, int(policy.get("max_active_model_instances", 1) or 1))
            except Exception:
                max_instances = 1
            if active_model_counts[-1] >= max_instances:
                saturated += 1

        avg_models = 0.0
        peak_models = 0
        if active_model_counts:
            avg_models = round(sum(active_model_counts) / len(active_model_counts), 3)
            peak_models = max(active_model_counts)
        avg_docs_coverage = (
            round(sum(docs_coverage_values) / len(docs_coverage_values), 6) if docs_coverage_values else 0.0
        )
        latest_docs_coverage = docs_coverage_values[-1] if docs_coverage_values else 0.0
        latest_docs_missing = docs_missing_values[-1] if docs_missing_values else 0
        top_tasks = [
            {"task": task, "count": count}
            for task, count in sorted(task_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:8]
        ]
        return {
            "contract": "eidos.runtime_trend_summary.v1",
            "count": len(entries),
            "average_active_models": avg_models,
            "peak_active_models": peak_models,
            "average_directory_docs_coverage": avg_docs_coverage,
            "latest_directory_docs_coverage": latest_docs_coverage,
            "latest_directory_docs_missing": latest_docs_missing,
            "peak_directory_docs_missing": max(docs_missing_values) if docs_missing_values else 0,
            "saturated_samples": saturated,
            "state_counts": state_counts,
            "top_tasks": top_tasks,
            "latest": entries[-1] if entries else {},
            "history": entries,
        }

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
            exclusive_owner = str(metadata.get("exclusive_owner") or "").strip()
            exclusive_active = bool(metadata.get("exclusive")) or bool(exclusive_owner)
            if allow_same_owner and current_owner and current_owner == str(owner):
                return {
                    "allowed": True,
                    "reason": "same_owner",
                    "active_model_count": len(active_models),
                    "max_active_model_instances": max_instances,
                    "max_active_model_families": max_families,
                }
            if exclusive_active and current_owner and current_owner != str(owner):
                return {
                    "allowed": False,
                    "reason": "exclusive_owner_active",
                    "active_model_count": len(active_models),
                    "active_owner": current_owner,
                    "exclusive_owner": exclusive_owner or current_owner,
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

    def recover_stale_owner(
        self,
        *,
        stale_after_sec: float = 1800.0,
        owner_filter: str | None = None,
        reason: str = "stale_owner_recovery",
    ) -> dict[str, Any]:
        with self._lock():
            payload = self._read_locked()
            owner = str(payload.get("owner") or "").strip()
            if not owner:
                return {"released": False, "reason": "no_owner", "payload": payload}
            if owner_filter and owner != str(owner_filter):
                return {"released": False, "reason": "owner_filter_mismatch", "payload": payload}
            metadata = dict(payload.get("metadata") or {})
            owner_pid = metadata.get("pid")
            if owner_pid and not _pid_alive(owner_pid):
                metadata.update(
                    {
                        "released_owner": owner,
                        "released_reason": f"{reason}_dead_pid",
                        "released_pid": int(owner_pid),
                    }
                )
                payload.update(
                    {
                        "owner": "",
                        "task": "idle",
                        "state": "idle",
                        "updated_at": _now_utc(),
                        "active_models": [],
                        "metadata": metadata,
                    }
                )
                self._save_locked(payload)
                self._append_history_locked(payload)
                return {"released": True, "reason": f"{reason}_dead_pid", "payload": payload}
            updated_at = _parse_iso_utc(payload.get("updated_at"))
            if updated_at is None:
                return {"released": False, "reason": "missing_updated_at", "payload": payload}
            age = (datetime.now(timezone.utc) - updated_at).total_seconds()
            if age < max(30.0, float(stale_after_sec)):
                return {"released": False, "reason": "not_stale", "age_sec": age, "payload": payload}
            metadata.update(
                {
                    "released_owner": owner,
                    "released_reason": str(reason),
                    "stale_age_sec": round(age, 3),
                }
            )
            payload.update(
                {
                    "owner": "",
                    "task": "idle",
                    "state": "idle",
                    "updated_at": _now_utc(),
                    "active_models": [],
                    "metadata": metadata,
                }
            )
            self._save_locked(payload)
            self._append_history_locked(payload)
            return {"released": True, "reason": str(reason), "age_sec": age, "payload": payload}

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
                "consecutive_failures": metadata.get("consecutive_failures"),
                "doc_model": metadata.get("doc_model"),
                "doc_thinking_mode": metadata.get("doc_thinking_mode"),
                "metadata": {
                    "directory_docs_missing": metadata.get("directory_docs_missing"),
                    "directory_docs_coverage": metadata.get("directory_docs_coverage"),
                    "directory_docs_missing_delta": metadata.get("directory_docs_missing_delta"),
                    "directory_docs_coverage_delta": metadata.get("directory_docs_coverage_delta"),
                },
                "summary": dict(metadata.get("summary") or {}) if isinstance(metadata.get("summary"), dict) else {},
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
