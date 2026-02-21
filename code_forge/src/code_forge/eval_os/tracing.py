from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_trace_id() -> str:
    return token_hex(16)


def new_span_id() -> str:
    return token_hex(8)


@dataclass(frozen=True)
class TraceMeta:
    run_id: str
    task_id: str
    config_id: str
    trace_id: str


class TraceRecorder:
    """
    Append-only JSONL event recorder for evaluation runs.

    Every event carries `trace_id` + `span_id` lineage and is flush-on-write
    for crash-safe partial traces.
    """

    def __init__(self, path: Path, *, meta: TraceMeta):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        event_type: str,
        span_id: str | None,
        parent_span_id: str | None,
        name: str,
        attributes: dict[str, Any] | None = None,
        status: str = "ok",
    ) -> None:
        payload = {
            "ts": utc_now_iso(),
            "event_type": str(event_type),
            "trace_id": self.meta.trace_id,
            "run_id": self.meta.run_id,
            "task_id": self.meta.task_id,
            "config_id": self.meta.config_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "name": str(name),
            "status": str(status),
            "attributes": dict(attributes or {}),
        }
        line = json.dumps(payload, sort_keys=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    @contextmanager
    def span(
        self,
        *,
        name: str,
        event_type: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        span_id = new_span_id()
        start = time.perf_counter()
        self.emit(
            event_type=f"{event_type}.start",
            span_id=span_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes or {},
            status="start",
        )
        status = "ok"
        err_repr = None
        try:
            yield span_id
        except Exception as exc:  # pragma: no cover - exercised through caller integration paths
            status = "error"
            err_repr = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            end_attrs = dict(attributes or {})
            end_attrs["duration_ms"] = elapsed_ms
            if err_repr:
                end_attrs["error"] = err_repr
            self.emit(
                event_type=f"{event_type}.end",
                span_id=span_id,
                parent_span_id=parent_span_id,
                name=name,
                attributes=end_attrs,
                status=status,
            )
