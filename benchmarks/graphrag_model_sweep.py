#!/usr/bin/env python3
"""Run GraphRAG bench + federated assessment across local candidate models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VENV_PYTHON = Path("eidosian_venv/bin/python")
BENCH_SCRIPT = Path("benchmarks/run_graphrag_bench.py")
ASSESS_SCRIPT = Path("benchmarks/graphrag_qualitative_assessor.py")
MODELS_DIR = Path("models")

@dataclass
class SweepResult:
    model_id: str
    model_path: str
    bench_ok: bool
    bench_returncode: int
    bench_stdout_tail: str
    metrics_path: str | None
    assessment_path: str | None
    final_score: float
    rank: str
    index_seconds: float | None
    query_seconds: float | None
    query_output: str


def _run(cmd: list[str], env: dict[str, str], timeout: int | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        return subprocess.CompletedProcess(cmd, 124, stdout=str(e.stdout or ""), stderr=str(e.stderr or ""))


def _latest_matching(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _score_tuple(result: SweepResult) -> tuple[float, float]:
    """Sort key: higher score first, then lower runtime."""
    runtime = (result.index_seconds or 9999.0) + (result.query_seconds or 9999.0)
    fail_penalty = -1.0 if not result.bench_ok else 0.0
    return (result.final_score + fail_penalty, -runtime)

def discover_models() -> list[tuple[str, str]]:
    specs: list[tuple[str, str]] = []
    if not MODELS_DIR.exists():
        return specs

    for model_path in MODELS_DIR.glob("*.gguf"):
        if "embed" in model_path.name.lower() or "mmproj" in model_path.name.lower():
            continue

        name = model_path.stem.lower().replace("-", "_").replace(".", "_")
        if name.endswith("_q8_0"):
            name = name[:-5]
        if name.endswith("_q6_k"):
            name = name[:-5]
        if name.endswith("_q4_k_m"):
            name = name[:-7]

        specs.append((name, str(model_path)))

    specs.sort(key=lambda x: x[0])
    return specs

def run_model(
    *,
    model_id: str,
    model_path: str,
    query: str,
    sweep_root: Path,
    skip_judges: bool,
    fallback_model_path: str,
) -> SweepResult:
    workspace_dir = sweep_root / model_id / "workspace"
    report_dir = sweep_root / model_id / "reports"
    workspace_dir.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["EIDOS_GRAPHRAG_WORKSPACE_DIR"] = str(workspace_dir)
    env["EIDOS_GRAPHRAG_REPORTS_DIR"] = str(report_dir)
    env["EIDOS_GRAPHRAG_INPUT_DIR"] = str(workspace_dir / "input_seed")
    env["EIDOS_GRAPHRAG_LLM_MODEL"] = model_path
    env["EIDOS_GRAPHRAG_LLM_MODEL_FALLBACK"] = fallback_model_path
    env["EIDOS_LLAMA_PARALLEL"] = "1"

    bench_cmd = [str(VENV_PYTHON), str(BENCH_SCRIPT), "--query", query]
    bench = _run(bench_cmd, env=env, timeout=None)
    bench_ok = bench.returncode == 0

    metrics_path = _latest_matching(report_dir, "bench_metrics_*.json")
    assessment_cmd = [
        str(VENV_PYTHON),
        str(ASSESS_SCRIPT),
        "--workspace-root",
        str(workspace_dir),
        "--report-dir",
        str(report_dir),
    ]
    if metrics_path:
        assessment_cmd.extend(["--metrics-json", str(metrics_path)])
    if skip_judges:
        assessment_cmd.append("--skip-judges")

    assess = _run(assessment_cmd, env=os.environ.copy(), timeout=600)
    assessment_path = _latest_matching(report_dir, "qualitative_assessment_*.json")
    assessment = _load_json(assessment_path)
    metrics = _load_json(metrics_path)

    aggregate = assessment.get("aggregate") if isinstance(assessment, dict) else {}
    if not isinstance(aggregate, dict):
        aggregate = {}

    return SweepResult(
        model_id=model_id,
        model_path=model_path,
        bench_ok=bench_ok,
        bench_returncode=bench.returncode,
        bench_stdout_tail="\n".join((bench.stdout or "").splitlines()[-20:]),
        metrics_path=str(metrics_path) if metrics_path else None,
        assessment_path=str(assessment_path) if assessment_path else None,
        final_score=float(aggregate.get("final_score", assessment.get("final_score", 0.0))),
        rank=str(aggregate.get("rank", assessment.get("rank", "N/A"))),
        index_seconds=float(metrics.get("index_seconds")) if metrics.get("index_seconds") is not None else None,
        query_seconds=float(metrics.get("query_seconds")) if metrics.get("query_seconds") is not None else None,
        query_output=str(metrics.get("query_output", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GraphRAG model sweep with federated qualitative scoring.")
    parser.add_argument(
        "--query",
        default="What is the relationship between Kael and Seraphina?",
        help="Evaluation query.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec 'name=path'. Repeatable. If not provided, discovers local GGUF models.",
    )
    parser.add_argument(
        "--skip-judges",
        action="store_true",
        help="Run deterministic-only assessments.",
    )
    parser.add_argument(
        "--output-root",
        default="reports/graphrag_sweep",
        help="Output root directory for sweep artifacts.",
    )
    args = parser.parse_args()

    specs: list[tuple[str, str]] = []
    if args.model:
        for raw in args.model:
            if "=" not in raw:
                raise SystemExit(f"Invalid --model value '{raw}'. Expected name=path.")
            name, path = raw.split("=", 1)
            specs.append((name.strip(), path.strip()))
    else:
        specs = discover_models()
        if not specs:
            raise SystemExit("No candidate GGUF models found in models/*.gguf")

    existing_model_paths = [path for _, path in specs if Path(path).exists()]
    fallback_model_path = existing_model_paths[0] if existing_model_paths else specs[0][1]

    sweep_root = Path(args.output_root).resolve()
    sweep_root.mkdir(parents=True, exist_ok=True)

    print(f"Starting sweep with {len(specs)} models...")
    for s in specs:
        print(f" - {s[0]}: {s[1]}")

    results: list[SweepResult] = []
    for model_id, model_path in specs:
        path_obj = Path(model_path)
        if not path_obj.exists():
            results.append(
                SweepResult(
                    model_id=model_id,
                    model_path=model_path,
                    bench_ok=False,
                    bench_returncode=127,
                    bench_stdout_tail="model not found",
                    metrics_path=None,
                    assessment_path=None,
                    final_score=0.0,
                    rank="N/A",
                    index_seconds=None,
                    query_seconds=None,
                    query_output="",
                )
            )
            continue
        
        print(f"\nRunning bench for: {model_id}...")
        result = run_model(
            model_id=model_id,
            model_path=model_path,
            query=args.query,
            sweep_root=sweep_root,
            skip_judges=args.skip_judges,
            fallback_model_path=fallback_model_path,
        )
        results.append(result)
        print(f"Result: ok={result.bench_ok}, score={result.final_score:.4f}, time={result.index_seconds}s")

    winner = max(results, key=_score_tuple)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": args.query,
        "skip_judges": bool(args.skip_judges),
        "winner": {
            "model_id": winner.model_id,
            "model_path": winner.model_path,
            "final_score": winner.final_score,
            "rank": winner.rank,
            "index_seconds": winner.index_seconds,
            "query_seconds": winner.query_seconds,
            "bench_ok": winner.bench_ok,
        },
        "results": [r.__dict__ for r in results],
    }
    summary_json = sweep_root / f"model_selection_{stamp}.json"
    summary_md = sweep_root / f"model_selection_{stamp}.md"
    summary_latest_json = sweep_root / "model_selection_latest.json"
    summary_latest_md = sweep_root / "model_selection_latest.md"
    summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    summary_latest_json.write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# GraphRAG Model Sweep",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Query: `{args.query}`",
        f"- Winner: `{winner.model_id}` ({winner.rank}, score={winner.final_score:.4f})",
        "",
        "## Results",
        "",
        "| Model | Bench OK | Rank | Score | Index s | Query s |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for r in results:
        lines.append(
            f"| `{r.model_id}` | `{r.bench_ok}` | `{r.rank}` | {r.final_score:.4f} | "
            f"{(r.index_seconds if r.index_seconds is not None else 'n/a')} | "
            f"{(r.query_seconds if r.query_seconds is not None else 'n/a')} |"
        )
    summary_md.write_text("\n".join(lines) + "\n")
    summary_latest_md.write_text("\n".join(lines) + "\n")

    print(f"\nSweep summary JSON: {summary_json}")
    print(f"Sweep summary Markdown: {summary_md}")
    print(f"Winner: {winner.model_id} (score={winner.final_score:.4f}, rank={winner.rank})")


if __name__ == "__main__":
    main()
