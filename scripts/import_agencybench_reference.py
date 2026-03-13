#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def aggregate_agencybench(root: Path) -> dict[str, Any]:
    scenario_files = sorted(root.glob("AgencyBench-v2/*/scenario*/claude/meta_eval.json"))
    if not scenario_files:
        raise FileNotFoundError(f"No AgencyBench sample meta_eval.json files found under {root}")
    scenarios = []
    total_subtasks = 0
    passed_subtasks = 0
    passed_scenarios = 0
    for path in scenario_files:
        payload = _load_json(path)
        subtasks = payload.get("subtasks") if isinstance(payload.get("subtasks"), list) else []
        subtask_total = len(subtasks)
        subtask_passed = sum(1 for item in subtasks if isinstance(item, dict) and bool(item.get("success")))
        category = path.parts[-4]
        scenario = path.parts[-3]
        success = subtask_total > 0 and subtask_passed == subtask_total
        if success:
            passed_scenarios += 1
        total_subtasks += subtask_total
        passed_subtasks += subtask_passed
        scenarios.append(
            {
                "category": category,
                "scenario": scenario,
                "path": str(path),
                "subtasks_total": subtask_total,
                "subtasks_passed": subtask_passed,
                "success": success,
            }
        )
    return {
        "contract": "eidos.external_benchmark_result.v1",
        "suite": "agencybench",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_path": str(root),
        "source_url": "https://github.com/GAIR-NLP/AgencyBench",
        "participant": "claude_sample",
        "execution_mode": "imported_reference",
        "status": "green" if passed_subtasks == total_subtasks else "yellow",
        "score": round(passed_subtasks / max(1, total_subtasks), 6),
        "metrics": {
            "success_rate": round(passed_subtasks / max(1, total_subtasks), 6),
            "scenario_pass_rate": round(passed_scenarios / max(1, len(scenarios)), 6),
            "tasks_total": len(scenarios),
            "tasks_passed": passed_scenarios,
            "subtasks_total": total_subtasks,
            "subtasks_passed": passed_subtasks,
        },
        "notes": "Aggregated official AgencyBench sample claude meta_eval.json artifacts.",
        "scenarios": scenarios,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate official AgencyBench sample results into the proof pipeline.")
    parser.add_argument("--agencybench-root", required=True, help="Path to a local AgencyBench checkout")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    root = Path(args.agencybench_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    payload = aggregate_agencybench(root)
    report_dir = repo_root / "reports" / "external_benchmarks" / "agencybench"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = report_dir / f"agencybench_{stamp}.json"
    latest_path = report_dir / "latest.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "latest": str(latest_path), "score": payload["score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
