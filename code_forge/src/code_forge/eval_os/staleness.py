from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


@dataclass(frozen=True)
class FreshnessRecord:
    memory_key: str
    source_last_modified: str
    derived_at: str
    served_at: str
    revalidated_at: str | None = None
    stale_error: bool = False
    metadata: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FreshnessRecord":
        for required in (
            "memory_key",
            "source_last_modified",
            "derived_at",
            "served_at",
        ):
            if not str(payload.get(required) or "").strip():
                raise ValueError(f"freshness record missing {required}")
        return cls(
            memory_key=str(payload["memory_key"]),
            source_last_modified=str(payload["source_last_modified"]),
            derived_at=str(payload["derived_at"]),
            served_at=str(payload["served_at"]),
            revalidated_at=(
                str(payload["revalidated_at"])
                if payload.get("revalidated_at")
                else None
            ),
            stale_error=bool(payload.get("stale_error", False)),
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_key": self.memory_key,
            "source_last_modified": self.source_last_modified,
            "derived_at": self.derived_at,
            "served_at": self.served_at,
            "revalidated_at": self.revalidated_at,
            "stale_error": self.stale_error,
            "metadata": dict(self.metadata or {}),
        }


def load_freshness_records(path: Path) -> list[FreshnessRecord]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            rows = list(payload.get("records") or [])
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(
                f"unsupported freshness payload type: {type(payload).__name__}"
            )
    return [FreshnessRecord.from_dict(row) for row in rows]


def compute_staleness_metrics(records: list[FreshnessRecord]) -> dict[str, Any]:
    if not records:
        return {
            "record_count": 0,
            "stale_serve_rate": 0.0,
            "staleness_caused_error_rate": 0.0,
            "freshness_lag_seconds": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
            "revalidation_latency_seconds": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
        }

    stale_serves = 0
    stale_errors = 0
    lag_values: list[float] = []
    revalidate_values: list[float] = []

    for row in records:
        source_ts = _parse_ts(row.source_last_modified)
        derived_ts = _parse_ts(row.derived_at)
        served_ts = _parse_ts(row.served_at)
        revalidated_ts = _parse_ts(row.revalidated_at)
        if not source_ts or not derived_ts or not served_ts:
            continue

        lag_values.append((served_ts - source_ts).total_seconds())

        is_stale = source_ts > derived_ts
        if is_stale:
            stale_serves += 1
            if row.stale_error:
                stale_errors += 1
            if revalidated_ts is not None:
                revalidate_values.append((revalidated_ts - served_ts).total_seconds())

    stale_serve_rate = stale_serves / max(1, len(records))
    stale_error_rate = stale_errors / max(1, stale_serves)
    lag_mean = statistics.mean(lag_values) if lag_values else 0.0
    revalidate_mean = statistics.mean(revalidate_values) if revalidate_values else 0.0

    return {
        "record_count": len(records),
        "stale_serve_count": stale_serves,
        "stale_error_count": stale_errors,
        "stale_serve_rate": stale_serve_rate,
        "staleness_caused_error_rate": stale_error_rate,
        "freshness_lag_seconds": {
            "mean": lag_mean,
            "p50": _percentile(lag_values, 50),
            "p95": _percentile(lag_values, 95),
        },
        "revalidation_latency_seconds": {
            "mean": revalidate_mean,
            "p50": _percentile(revalidate_values, 50),
            "p95": _percentile(revalidate_values, 95),
        },
    }
