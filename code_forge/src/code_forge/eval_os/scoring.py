from __future__ import annotations

import statistics
from typing import Any


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    idx = int(round((pct / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


def score_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {
            "run_count": 0,
            "success_rate": 0.0,
            "latency_ms": {"mean": 0.0, "p50": 0.0, "p95": 0.0},
            "variance": {"success_stddev": 0.0, "latency_stddev": 0.0},
            "quality_gate": {"regression_rate": 0.0},
            "enterprise_score": 0.0,
        }

    success_values = [1.0 if bool(r.get("success")) else 0.0 for r in runs]
    latency_values = [float(r.get("duration_ms") or 0.0) for r in runs]
    regression_values = [1.0 if bool(r.get("regression")) else 0.0 for r in runs]
    stale_error_values = [
        float(((r.get("staleness") or {}).get("staleness_caused_error_rate") or 0.0))
        for r in runs
    ]

    success_rate = statistics.mean(success_values)
    latency_mean = statistics.mean(latency_values) if latency_values else 0.0
    regression_rate = statistics.mean(regression_values) if regression_values else 0.0
    stale_error_rate = (
        statistics.mean(stale_error_values) if stale_error_values else 0.0
    )

    # Weighted objective with explicit trade-offs.
    # 1. reward task success strongly
    # 2. penalize regressions + stale errors
    # 3. mildly penalize latency inflation
    latency_penalty = min(1.0, latency_mean / 600_000.0)
    enterprise_score = (
        (0.70 * success_rate)
        - (0.15 * regression_rate)
        - (0.10 * stale_error_rate)
        - (0.05 * latency_penalty)
    )
    enterprise_score = max(0.0, min(1.0, enterprise_score))

    return {
        "run_count": len(runs),
        "success_rate": success_rate,
        "latency_ms": {
            "mean": latency_mean,
            "p50": _percentile(latency_values, 50),
            "p95": _percentile(latency_values, 95),
        },
        "variance": {
            "success_stddev": (
                statistics.pstdev(success_values) if len(success_values) > 1 else 0.0
            ),
            "latency_stddev": (
                statistics.pstdev(latency_values) if len(latency_values) > 1 else 0.0
            ),
        },
        "quality_gate": {
            "regression_rate": regression_rate,
            "staleness_caused_error_rate": stale_error_rate,
        },
        "enterprise_score": enterprise_score,
    }
