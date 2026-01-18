"""Schema versioning and migrations."""

from __future__ import annotations

from typing import Any

CURRENT_SCHEMA_VERSION = 3


def migrate_document_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Upgrade an index document dict to the current schema version."""

    if "schema_version" not in payload:
        payload = {**payload, "schema_version": 1}

    version = int(payload.get("schema_version", 0))
    if version == CURRENT_SCHEMA_VERSION:
        return payload
    if version == 1:
        payload = _migrate_v1_to_v2(payload)
        version = 2
    if version == 2:
        return _migrate_v2_to_v3(payload)

    raise ValueError(f"Unsupported schema version: {version}")


def _migrate_v1_to_v2(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate schema v1 payloads to v2."""

    return {
        **payload,
        "schema_version": 2,
        "test_summary": payload.get("test_summary"),
        "profile_summary": payload.get("profile_summary"),
        "benchmark_summary": payload.get("benchmark_summary"),
    }


def _migrate_v2_to_v3(payload: dict[str, Any]) -> dict[str, Any]:
    """Migrate schema v2 payloads to v3."""

    benchmark_summary = payload.get("benchmark_summary")
    if isinstance(benchmark_summary, dict) and "benchmarks" not in benchmark_summary:
        benchmark_summary = {
            "benchmarks": [
                {
                    "name": "benchmark",
                    "runs": benchmark_summary.get("runs", 0),
                    "mean_seconds": benchmark_summary.get("mean_seconds", 0.0),
                    "median_seconds": benchmark_summary.get("median_seconds", 0.0),
                    "stdev_seconds": benchmark_summary.get("stdev_seconds", 0.0),
                    "min_seconds": benchmark_summary.get("min_seconds", 0.0),
                    "max_seconds": benchmark_summary.get("max_seconds", 0.0),
                }
            ]
        }
    return {
        **payload,
        "schema_version": 3,
        "benchmark_summary": benchmark_summary,
    }
