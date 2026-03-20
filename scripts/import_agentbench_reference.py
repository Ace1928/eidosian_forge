#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cell(row: list[str], idx: int) -> str:
    if idx < 0 or idx >= len(row):
        return ""
    return str(row[idx]).strip()


def aggregate_agentbench(csv_path: Path) -> dict[str, Any]:
    rows = list(csv.reader(csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()))
    entries: list[dict[str, Any]] = []
    for row in rows:
        if not row or len(row) < 14:
            continue
        model = _cell(row, 0)
        if not model or model.lower() == "model" or model.lower().startswith("email "):
            continue
        entry = {
            "model": model,
            "organization": _cell(row, 1),
            "result_source": _cell(row, 2),
            "release_date": _cell(row, 3),
            "alfworld": _safe_float(_cell(row, 4)),
            "db": _safe_float(_cell(row, 6)),
            "kg": _safe_float(_cell(row, 8)),
            "os": _safe_float(_cell(row, 10)),
            "webshop": _safe_float(_cell(row, 12)),
            "avg": _safe_float(_cell(row, 14)),
        }
        if entry["avg"] <= 0.0:
            continue
        entries.append(entry)

    if not entries:
        raise ValueError(f"No AgentBench leaderboard rows found in {csv_path}")

    ranked = sorted(entries, key=lambda item: item["avg"], reverse=True)
    best = ranked[0]
    open_entries = [
        row
        for row in ranked
        if not any(token in row["model"].lower() for token in ("gpt-", "claude", "gemini", "o1", "o3"))
    ]
    best_open = open_entries[0] if open_entries else best

    return {
        "contract": "eidos.external_benchmark_result.v1",
        "suite": "agentbench",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_path": str(csv_path),
        "source_url": "https://github.com/THUDM/AgentBench",
        "participant": "agentbench_leaderboard_reference",
        "execution_mode": "imported_reference",
        "status": "green" if best["avg"] >= 80.0 else "yellow" if best["avg"] >= 50.0 else "red",
        "score": round(best["avg"] / 100.0, 6),
        "metrics": {
            "leaderboard_best_avg": round(best["avg"], 6),
            "open_model_best_avg": round(best_open["avg"], 6),
            "tasks_total": 5,
            "rows_total": len(ranked),
        },
        "notes": "Aggregated official AgentBench leaderboard CSV published from the official repository README.",
        "leaderboard": {
            "best_model": best["model"],
            "best_organization": best["organization"],
            "best_result_source": best["result_source"],
            "best_release_date": best["release_date"],
            "best_open_model": best_open["model"],
        },
        "entries": ranked[:25],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate official AgentBench leaderboard CSV into proof artifacts.")
    parser.add_argument("--csv", required=True, help="Path to a downloaded AgentBench leaderboard CSV")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    payload = aggregate_agentbench(Path(args.csv).resolve())
    report_dir = repo_root / "reports" / "external_benchmarks" / "agentbench"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = report_dir / f"agentbench_{stamp}.json"
    latest_path = report_dir / "latest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "latest": str(latest_path), "score": payload["score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
