#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _pick_metric(payload: dict[str, Any], keys: list[str]) -> float:
    for key in keys:
        if key in payload:
            return _safe_float(payload.get(key))
    for container_name in ("metrics", "summary", "aggregate", "results"):
        container = payload.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in keys:
            if key in container:
                return _safe_float(container.get(key))
    return 0.0


def normalize_external_benchmark(
    *,
    suite: str,
    input_path: Path,
    source_url: str = "",
    notes: str = "",
) -> dict[str, Any]:
    payload = _load_json(input_path)
    score = _pick_metric(payload, ["score", "success_rate", "pass_rate", "resolved_rate"])
    success_rate = _pick_metric(payload, ["success_rate", "pass_rate", "resolved_rate", "score"])
    tasks_total = int(_pick_metric(payload, ["tasks_total", "instances_total", "total"]))
    tasks_passed = int(_pick_metric(payload, ["tasks_passed", "resolved", "passed", "successes"]))
    status = "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red"
    return {
        "contract": "eidos.external_benchmark_result.v1",
        "suite": suite.lower(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_path": str(input_path),
        "source_url": source_url,
        "status": status,
        "score": round(score, 6),
        "metrics": {
            "score": round(score, 6),
            "success_rate": round(success_rate, 6),
            "tasks_total": tasks_total,
            "tasks_passed": tasks_passed,
        },
        "notes": notes,
        "raw_excerpt": {key: payload.get(key) for key in sorted(payload)[:12]},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize external benchmark evidence into Eidos proof artifacts.")
    parser.add_argument("--suite", required=True, help="Benchmark suite name, e.g. agentbench or swebench")
    parser.add_argument("--input", required=True, help="Path to the upstream benchmark JSON summary")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--source-url", default="", help="Upstream suite URL or results page URL")
    parser.add_argument("--notes", default="", help="Free-form operator notes")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_path = Path(args.input).resolve()
    report = normalize_external_benchmark(
        suite=args.suite,
        input_path=input_path,
        source_url=args.source_url,
        notes=args.notes,
    )
    report_dir = repo_root / "reports" / "external_benchmarks" / report["suite"]
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = report_dir / f"{report['suite']}_{stamp}.json"
    latest_path = report_dir / "latest.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "latest": str(latest_path), "score": report["score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
