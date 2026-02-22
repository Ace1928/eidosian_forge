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
from urllib import error as urlerror
from urllib import request as urlrequest


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


def _parse_ts_ns(value: str) -> int:
    text = str(value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1_000_000_000)


def _safe_span_hex(value: str | None, size: int) -> str:
    if not value:
        return token_hex(size // 2)
    text = str(value).strip().lower()
    valid = len(text) == size and all(ch in "0123456789abcdef" for ch in text)
    return text if valid else token_hex(size // 2)


def _to_otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def _attrs_to_otlp(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted(attrs.keys()):
        out.append({"key": str(key), "value": _to_otlp_value(attrs[key])})
    return out


def build_otlp_trace_payload(
    events: list[dict[str, Any]],
    *,
    service_name: str,
    resource_attributes: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], int]:
    spans: list[dict[str, Any]] = []
    for row in events:
        trace_id = _safe_span_hex(str(row.get("trace_id") or ""), 32)
        span_id = _safe_span_hex(row.get("span_id"), 16)
        parent_id_raw = row.get("parent_span_id")
        parent_span_id = _safe_span_hex(parent_id_raw, 16) if parent_id_raw else None
        event_ts = _parse_ts_ns(str(row.get("ts") or utc_now_iso()))
        attrs = dict(row.get("attributes") or {})
        attrs.setdefault("event_type", str(row.get("event_type") or "event"))
        attrs.setdefault("status", str(row.get("status") or "ok"))
        attrs.setdefault("run_id", str(row.get("run_id") or ""))
        attrs.setdefault("task_id", str(row.get("task_id") or ""))
        attrs.setdefault("config_id", str(row.get("config_id") or ""))
        duration_ms = attrs.get("duration_ms")
        start_ns = event_ts
        end_ns = event_ts
        if duration_ms is not None:
            try:
                dur_ns = max(1, int(float(duration_ms) * 1_000_000))
                start_ns = max(0, event_ts - dur_ns)
            except (TypeError, ValueError):
                pass
        if end_ns <= start_ns:
            end_ns = start_ns + 1
        span_payload: dict[str, Any] = {
            "traceId": trace_id,
            "spanId": span_id,
            "name": str(row.get("name") or attrs.get("event_type") or "event"),
            "startTimeUnixNano": str(start_ns),
            "endTimeUnixNano": str(end_ns),
            "attributes": _attrs_to_otlp(attrs),
        }
        if parent_span_id:
            span_payload["parentSpanId"] = parent_span_id
        spans.append(span_payload)

    resource_attrs = {"service.name": service_name}
    resource_attrs.update(resource_attributes or {})
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs_to_otlp(resource_attrs)},
                "scopeSpans": [{"scope": {"name": "code_forge.eval_os"}, "spans": spans}],
            }
        ]
    }
    return payload, len(spans)


def _normalize_otlp_endpoint(endpoint: str) -> str:
    base = str(endpoint or "").strip()
    if not base:
        raise ValueError("OTLP endpoint is required")
    return base.rstrip("/") if base.rstrip("/").endswith("/v1/traces") else f"{base.rstrip('/')}/v1/traces"


def export_trace_jsonl_to_otlp(
    *,
    trace_path: Path,
    endpoint: str,
    service_name: str = "code_forge_eval",
    headers: dict[str, str] | None = None,
    timeout_sec: int = 10,
    resource_attributes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    trace_path = Path(trace_path).resolve()
    rows: list[dict[str, Any]] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    payload, span_count = build_otlp_trace_payload(
        rows,
        service_name=service_name,
        resource_attributes=resource_attributes,
    )
    target = _normalize_otlp_endpoint(endpoint)
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    body = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(target, data=body, method="POST", headers=request_headers)
    try:
        with urlrequest.urlopen(req, timeout=max(1, int(timeout_sec))) as resp:
            status = int(getattr(resp, "status", 0) or resp.getcode() or 0)
            response_text = resp.read().decode("utf-8", errors="replace")
        ok = 200 <= status < 300
        return {
            "ok": ok,
            "endpoint": target,
            "status_code": status,
            "events": len(rows),
            "spans": span_count,
            "response": response_text[:2000],
        }
    except (urlerror.URLError, TimeoutError, ValueError) as exc:
        return {
            "ok": False,
            "endpoint": target,
            "status_code": None,
            "events": len(rows),
            "spans": span_count,
            "error": f"{type(exc).__name__}: {exc}",
        }
