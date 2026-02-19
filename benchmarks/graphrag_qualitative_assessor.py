#!/usr/bin/env python3
"""Federated qualitative assessment for GraphRAG pipeline outputs.

This tool combines deterministic measurement contracts with local multi-model
judges (OpenAI-compatible llama-server endpoints) and emits a reproducible
multi-dimensional scorecard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEFAULT_WORKSPACE_ROOT = Path("data/graphrag_test/workspace")
DEFAULT_REPORT_DIR = Path("reports/graphrag")
DEFAULT_SCHEMA_PATH = Path("benchmarks/schemas/graphrag_qualitative_assessment.schema.json")
DEFAULT_LLAMA_SERVER_BIN = Path("llama.cpp/build/bin/llama-server")
DEFAULT_MODELS = [
    "qwen=models/Qwen2.5-0.5B-Instruct-Q8_0.gguf",
    "llama=models/Llama-3.2-1B-Instruct-Q8_0.gguf",
]

EXPECTED_WORKFLOWS = [
    "load_input_documents",
    "create_base_text_units",
    "create_final_documents",
    "extract_graph_nlp",
    "prune_graph",
    "finalize_graph",
    "create_communities",
    "create_final_text_units",
    "create_community_reports_text",
]
EXPECTED_ENTITIES = ["ALARIC", "SERAPHINA", "KAEL", "MALAKAR", "CRYSTAL", "WHISPERING CAVES", "EIDOS"]
PLACEHOLDER_MARKERS = ["auto-generated placeholder", "fallback generated"]
CONTRACT_VERSION = "graphrag.qualitative.assessment.v1"


@dataclass
class JudgeSpec:
    name: str
    model_path: Path
    port: int


@dataclass
class JudgeRuntime:
    spec: JudgeSpec
    process: subprocess.Popen[Any]
    log_file: Any


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _llama_env(llama_server_bin: Path) -> dict[str, str]:
    env = os.environ.copy()
    bin_dir = str(llama_server_bin.resolve().parent)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    ld_library = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_library}" if ld_library else bin_dir
    return env


def _wait_for_http(url: str, timeout_s: float = 40.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")


def _http_json(url: str, payload: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _extract_json_fragment(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _find_latest_metrics(report_dir: Path) -> Path | None:
    candidates = sorted(report_dir.glob("bench_metrics_*.json"))
    return candidates[-1] if candidates else None


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_artifacts(workspace_root: Path, metrics_path: Path | None) -> dict[str, Any]:
    output_dir = workspace_root / "output"
    stats_path = output_dir / "stats.json"
    entities_path = output_dir / "entities.parquet"
    relationships_path = output_dir / "relationships.parquet"
    community_reports_path = output_dir / "community_reports.parquet"

    artifacts: dict[str, Any] = {
        "workspace_root": str(workspace_root),
        "output_dir": str(output_dir),
        "required_files": {
            "stats": str(stats_path),
            "entities": str(entities_path),
            "relationships": str(relationships_path),
            "community_reports": str(community_reports_path),
        },
        "missing_files": [],
    }

    for name, path in (
        ("stats", stats_path),
        ("entities", entities_path),
        ("relationships", relationships_path),
        ("community_reports", community_reports_path),
    ):
        if not path.exists():
            artifacts["missing_files"].append(name)

    if stats_path.exists():
        artifacts["stats"] = json.loads(stats_path.read_text())

    if entities_path.exists():
        entities_df = pd.read_parquet(entities_path)
        artifacts["entities_count"] = int(len(entities_df))
        titles = [str(v).upper() for v in entities_df.get("title", [])]
        artifacts["entity_titles"] = titles
        artifacts["expected_entity_hits"] = sorted({token for token in EXPECTED_ENTITIES if any(token in t for t in titles)})
    else:
        artifacts["entities_count"] = 0
        artifacts["entity_titles"] = []
        artifacts["expected_entity_hits"] = []

    if relationships_path.exists():
        relationships_df = pd.read_parquet(relationships_path)
        artifacts["relationships_count"] = int(len(relationships_df))
    else:
        artifacts["relationships_count"] = 0

    if community_reports_path.exists():
        reports_df = pd.read_parquet(community_reports_path)
        artifacts["community_reports_count"] = int(len(reports_df))
        summaries = [str(v).lower() for v in reports_df.get("summary", [])]
        artifacts["community_reports_placeholder"] = any(
            marker in summary for summary in summaries for marker in PLACEHOLDER_MARKERS
        )
        artifacts["community_summaries"] = [str(v) for v in reports_df.get("summary", [])][:5]
    else:
        artifacts["community_reports_count"] = 0
        artifacts["community_reports_placeholder"] = True
        artifacts["community_summaries"] = []

    artifacts["bench_metrics_path"] = str(metrics_path) if metrics_path else None
    artifacts["bench_metrics"] = {}
    artifacts["query_output"] = ""
    artifacts["index_seconds"] = None
    artifacts["query_seconds"] = None

    if metrics_path and metrics_path.exists():
        bench = json.loads(metrics_path.read_text())
        artifacts["bench_metrics"] = bench
        artifacts["query_output"] = str(bench.get("query_output") or "")
        artifacts["index_seconds"] = _to_float(bench.get("index_seconds"), default=-1.0)
        artifacts["query_seconds"] = _to_float(bench.get("query_seconds"), default=-1.0)

    return artifacts


def deterministic_scores(artifacts: dict[str, Any]) -> dict[str, float]:
    stats = artifacts.get("stats") or {}
    workflows = stats.get("workflows") or {}
    completed_workflows = [wf for wf in EXPECTED_WORKFLOWS if wf in workflows]

    pipeline_integrity = _clamp01(1.0 - _safe_ratio(len(artifacts.get("missing_files", [])), 4))
    workflow_completeness = _clamp01(_safe_ratio(len(completed_workflows), len(EXPECTED_WORKFLOWS)))
    entity_coverage = _clamp01(_safe_ratio(len(artifacts.get("expected_entity_hits", [])), len(EXPECTED_ENTITIES)))

    entities_count = max(1, int(artifacts.get("entities_count", 0)))
    relationships_count = int(artifacts.get("relationships_count", 0))
    relationship_density = _clamp01(_safe_ratio(relationships_count, entities_count * 6))

    community_reports_count = int(artifacts.get("community_reports_count", 0))
    placeholder = bool(artifacts.get("community_reports_placeholder", True))
    community_report_quality = 0.0
    if community_reports_count > 0:
        community_report_quality = 0.35 if placeholder else 0.9

    query_output = (artifacts.get("query_output") or "").strip()
    query_answer_quality = 0.1 if not query_output else 0.65
    query_upper = query_output.upper()
    if "KAEL" in query_upper and "SERAPHINA" in query_upper:
        query_answer_quality = 0.95

    index_seconds = artifacts.get("index_seconds")
    query_seconds = artifacts.get("query_seconds")
    runtime_score = 0.5
    if isinstance(index_seconds, (int, float)) and index_seconds > 0 and isinstance(query_seconds, (int, float)) and query_seconds > 0:
        index_component = _clamp01(1.0 - (index_seconds / 120.0))
        query_component = _clamp01(1.0 - (query_seconds / 20.0))
        runtime_score = round((0.7 * index_component + 0.3 * query_component), 4)

    return {
        "pipeline_integrity": round(pipeline_integrity, 4),
        "workflow_completeness": round(workflow_completeness, 4),
        "entity_coverage": round(entity_coverage, 4),
        "relationship_density": round(relationship_density, 4),
        "community_report_quality": round(community_report_quality, 4),
        "query_answer_quality": round(query_answer_quality, 4),
        "runtime_score": round(runtime_score, 4),
    }


def _build_stage_scores(artifacts: dict[str, Any]) -> list[dict[str, Any]]:
    stats = artifacts.get("stats") or {}
    workflows = stats.get("workflows") or {}
    stage_scores = []
    for wf in EXPECTED_WORKFLOWS:
        record = workflows.get(wf) or {}
        runtime = _to_float(record.get("overall"), default=-1.0)
        completed = wf in workflows
        stage_score = 0.0
        if completed and runtime >= 0:
            stage_score = _clamp01(1.0 - (runtime / 45.0))
        stage_scores.append(
            {
                "stage": wf,
                "completed": completed,
                "runtime_seconds": runtime,
                "stage_score": round(stage_score, 4),
            }
        )
    return stage_scores


def parse_judge_specs(models: Iterable[str], port_base: int) -> list[JudgeSpec]:
    specs: list[JudgeSpec] = []
    port = port_base
    for raw in models:
        if "=" in raw:
            name, model = raw.split("=", 1)
        else:
            model_path = Path(raw)
            name = model_path.stem
            model = raw
        model_path = Path(model)
        specs.append(JudgeSpec(name=name.strip(), model_path=model_path, port=port))
        port += 1
    return specs


def start_judges(specs: list[JudgeSpec], llama_server_bin: Path, report_dir: Path) -> list[JudgeRuntime]:
    report_dir.mkdir(parents=True, exist_ok=True)
    runtimes: list[JudgeRuntime] = []
    for spec in specs:
        if not spec.model_path.exists():
            continue
        log_path = report_dir / f"judge_{spec.name}_{spec.port}.log"
        log_handle = log_path.open("w")
        cmd = [
            str(llama_server_bin),
            "-m",
            str(spec.model_path),
            "--port",
            str(spec.port),
            "--ctx-size",
            "2048",
            "--n-gpu-layers",
            os.environ.get("EIDOS_LLAMA_GPU_LAYERS", "0"),
            "--parallel",
            "2",
        ]
        process = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, env=_llama_env(llama_server_bin))
        try:
            _wait_for_http(f"http://127.0.0.1:{spec.port}/health", timeout_s=60.0)
        except Exception:
            process.terminate()
            log_handle.close()
            continue
        runtimes.append(JudgeRuntime(spec=spec, process=process, log_file=log_handle))
    return runtimes


def stop_judges(runtimes: list[JudgeRuntime]) -> None:
    for runtime in runtimes:
        try:
            runtime.process.terminate()
        except Exception:
            pass
    for runtime in runtimes:
        try:
            runtime.process.wait(timeout=8)
        except Exception:
            try:
                runtime.process.kill()
            except Exception:
                pass
        try:
            runtime.log_file.close()
        except Exception:
            pass


def _judge_prompt(artifacts: dict[str, Any], deterministic: dict[str, float], stage_scores: list[dict[str, Any]]) -> str:
    payload = {
        "contract_version": CONTRACT_VERSION,
        "summary": {
            "entities_count": artifacts.get("entities_count"),
            "relationships_count": artifacts.get("relationships_count"),
            "community_reports_placeholder": artifacts.get("community_reports_placeholder"),
            "expected_entity_hits": artifacts.get("expected_entity_hits"),
            "query_output": (artifacts.get("query_output") or "")[:1200],
        },
        "deterministic_scores": deterministic,
        "stage_scores": stage_scores,
    }
    return (
        "Assess GraphRAG output quality. Return ONLY strict JSON with keys: "
        "scores (factuality, grounding, coherence, usefulness, risk_awareness each 0..1), "
        "verdict (one short sentence), and risks (array of short strings).\n"
        "Do not use markdown.\n"
        f"DATA:\n{json.dumps(payload, ensure_ascii=True)}"
    )


def run_judges(runtimes: list[JudgeRuntime], artifacts: dict[str, Any], deterministic: dict[str, float], stage_scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    prompt = _judge_prompt(artifacts, deterministic, stage_scores)
    for runtime in runtimes:
        url = f"http://127.0.0.1:{runtime.spec.port}/v1/chat/completions"
        req = {
            "model": "local",
            "temperature": 0.0,
            "max_tokens": 512,
            "messages": [
                {"role": "system", "content": "You are a strict evaluator. Output valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        raw: dict[str, Any] | None = None
        parsed: dict[str, Any] | None = None
        err: str | None = None
        try:
            raw = _http_json(url, req, timeout_s=120.0)
            text = (
                raw.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            parsed = _extract_json_fragment(text)
            if parsed is None:
                raise ValueError("judge returned non-JSON content")
        except Exception as exc:
            err = str(exc)

        score_block = (parsed or {}).get("scores") if isinstance(parsed, dict) else {}
        entry = {
            "judge": runtime.spec.name,
            "model_path": str(runtime.spec.model_path),
            "port": runtime.spec.port,
            "scores": {
                "factuality": _clamp01(_to_float((score_block or {}).get("factuality"), 0.0)),
                "grounding": _clamp01(_to_float((score_block or {}).get("grounding"), 0.0)),
                "coherence": _clamp01(_to_float((score_block or {}).get("coherence"), 0.0)),
                "usefulness": _clamp01(_to_float((score_block or {}).get("usefulness"), 0.0)),
                "risk_awareness": _clamp01(_to_float((score_block or {}).get("risk_awareness"), 0.0)),
            },
            "verdict": (parsed or {}).get("verdict", ""),
            "risks": (parsed or {}).get("risks", []),
            "error": err,
            "valid": err is None,
            "raw_response": raw,
        }
        assessments.append(entry)
    return assessments


def aggregate_scores(deterministic: dict[str, float], judge_assessments: list[dict[str, Any]]) -> dict[str, Any]:
    deterministic_objective = (
        0.20 * deterministic["pipeline_integrity"]
        + 0.10 * deterministic["workflow_completeness"]
        + 0.15 * deterministic["entity_coverage"]
        + 0.10 * deterministic["relationship_density"]
        + 0.15 * deterministic["community_report_quality"]
        + 0.20 * deterministic["query_answer_quality"]
        + 0.10 * deterministic["runtime_score"]
    )

    judge_dimensions = ["factuality", "grounding", "coherence", "usefulness", "risk_awareness"]
    per_dim: dict[str, list[float]] = {key: [] for key in judge_dimensions}
    valid_assessments = [a for a in judge_assessments if a.get("valid", True)]
    for assessment in valid_assessments:
        scores = assessment.get("scores") or {}
        for dim in judge_dimensions:
            per_dim[dim].append(_clamp01(_to_float(scores.get(dim), 0.0)))

    judge_median = {
        dim: (statistics.median(values) if values else 0.0)
        for dim, values in per_dim.items()
    }
    judge_spread = {
        dim: (max(values) - min(values) if values else 0.0)
        for dim, values in per_dim.items()
    }
    judge_score = sum(judge_median.values()) / len(judge_dimensions)
    if not valid_assessments:
        judge_score = deterministic_objective
        judge_median = {dim: deterministic_objective for dim in judge_dimensions}
        judge_spread = {dim: 0.0 for dim in judge_dimensions}

    disagreement = 0.0
    all_totals: list[float] = []
    for assessment in valid_assessments:
        scores = assessment.get("scores") or {}
        total = sum(_clamp01(_to_float(scores.get(dim), 0.0)) for dim in judge_dimensions) / len(judge_dimensions)
        all_totals.append(total)
    if len(all_totals) > 1:
        disagreement = statistics.pstdev(all_totals)

    consensus_penalty = _clamp01(disagreement)
    base_score = _clamp01((0.75 * deterministic_objective + 0.25 * judge_score) * (1.0 - 0.25 * consensus_penalty))
    guardrail_flags: list[str] = []
    if deterministic.get("community_report_quality", 0.0) < 0.5:
        guardrail_flags.append("community_report_placeholder")
    if deterministic.get("query_answer_quality", 0.0) < 0.5:
        guardrail_flags.append("weak_query_answer")
    if deterministic.get("pipeline_integrity", 0.0) < 1.0:
        guardrail_flags.append("pipeline_integrity_gap")
    if deterministic.get("workflow_completeness", 0.0) < 1.0:
        guardrail_flags.append("workflow_incomplete")
    if judge_assessments and not valid_assessments:
        guardrail_flags.append("all_judges_failed")

    final_score = base_score
    if guardrail_flags:
        final_score = min(final_score, 0.79)

    rank = "A"
    if final_score < 0.8:
        rank = "B"
    if final_score < 0.6:
        rank = "C"
    if final_score < 0.4:
        rank = "D"

    return {
        "deterministic_objective": round(deterministic_objective, 4),
        "judge_score": round(judge_score, 4),
        "judge_dimension_median": {k: round(v, 4) for k, v in judge_median.items()},
        "judge_dimension_spread": {k: round(v, 4) for k, v in judge_spread.items()},
        "disagreement": round(disagreement, 4),
        "consensus_penalty": round(consensus_penalty, 4),
        "base_score": round(base_score, 4),
        "final_score": round(final_score, 4),
        "rank": rank,
        "guardrail_flags": guardrail_flags,
    }


def validate_contract(report: dict[str, Any]) -> None:
    required_top = {
        "contract_version",
        "generated_at",
        "workspace_root",
        "deterministic_scores",
        "stage_scores",
        "judge_assessments",
        "aggregate",
    }
    missing = sorted(required_top - set(report.keys()))
    if missing:
        raise ValueError(f"Assessment report missing required keys: {missing}")


def write_reports(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"qualitative_assessment_{stamp}.json"
    md_path = report_dir / f"qualitative_assessment_{stamp}.md"

    json_path.write_text(json.dumps(report, indent=2) + "\n")

    agg = report["aggregate"]
    lines = [
        "# GraphRAG Qualitative Assessment",
        "",
        f"- Contract: `{report['contract_version']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Workspace: `{report['workspace_root']}`",
        f"- Final score: **{agg['final_score']:.4f}** (rank `{agg['rank']}`)",
        f"- Deterministic objective: `{agg['deterministic_objective']:.4f}`",
        f"- Judge score: `{agg['judge_score']:.4f}`",
        f"- Disagreement: `{agg['disagreement']:.4f}`",
        f"- Guardrails: `{', '.join(agg.get('guardrail_flags', [])) or 'none'}`",
        "",
        "## Deterministic Scores",
    ]
    for key, value in report["deterministic_scores"].items():
        lines.append(f"- `{key}`: `{value:.4f}`")

    lines.append("")
    lines.append("## Stage Scores")
    for stage in report["stage_scores"]:
        lines.append(
            f"- `{stage['stage']}`: completed=`{stage['completed']}` runtime=`{stage['runtime_seconds']}` score=`{stage['stage_score']}`"
        )

    lines.append("")
    lines.append("## Judge Consensus Spread")
    spread = agg.get("judge_dimension_spread", {})
    if spread:
        for key, value in spread.items():
            lines.append(f"- `{key}` spread: `{value:.4f}`")
    else:
        lines.append("- No spread metrics available.")

    lines.append("")
    lines.append("## Judge Assessments")
    if report["judge_assessments"]:
        for assessment in report["judge_assessments"]:
            scores = assessment.get("scores", {})
            lines.append(
                f"- `{assessment['judge']}`: factuality={scores.get('factuality')} grounding={scores.get('grounding')} coherence={scores.get('coherence')} usefulness={scores.get('usefulness')} risk_awareness={scores.get('risk_awareness')}"
            )
            if assessment.get("error"):
                lines.append(f"  error: {assessment['error']}")
    else:
        lines.append("- No judge assessments (all model judges skipped or failed).")

    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Federated qualitative assessment for GraphRAG output artifacts.")
    parser.add_argument("--workspace-root", default=str(DEFAULT_WORKSPACE_ROOT), help="GraphRAG workspace root.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Report output directory.")
    parser.add_argument("--metrics-json", default="", help="Optional benchmark metrics JSON path.")
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="Schema file path used for contract metadata.")
    parser.add_argument("--llama-server-bin", default=str(DEFAULT_LLAMA_SERVER_BIN), help="Path to llama-server binary.")
    parser.add_argument("--judge-model", action="append", default=[], help="Judge model spec name=path (repeatable).")
    parser.add_argument("--port-base", type=int, default=8091, help="Base port for judge servers.")
    parser.add_argument("--skip-judges", action="store_true", help="Skip local model judges and run deterministic scoring only.")
    args = parser.parse_args()

    workspace_root = Path(args.workspace_root)
    report_dir = Path(args.report_dir)
    metrics_path = Path(args.metrics_json) if args.metrics_json else _find_latest_metrics(report_dir)

    artifacts = load_artifacts(workspace_root, metrics_path)
    deterministic = deterministic_scores(artifacts)
    stage_scores = _build_stage_scores(artifacts)

    judge_specs = parse_judge_specs(args.judge_model or DEFAULT_MODELS, args.port_base)
    judge_assessments: list[dict[str, Any]] = []

    runtimes: list[JudgeRuntime] = []
    if not args.skip_judges:
        llama_server_bin = Path(args.llama_server_bin)
        if not llama_server_bin.exists():
            print(f"[warn] llama-server not found at {llama_server_bin}; skipping judges", file=sys.stderr)
        else:
            runtimes = start_judges(judge_specs, llama_server_bin, report_dir)
            try:
                judge_assessments = run_judges(runtimes, artifacts, deterministic, stage_scores)
            finally:
                stop_judges(runtimes)

    aggregate = aggregate_scores(deterministic, judge_assessments)

    report = {
        "contract_version": CONTRACT_VERSION,
        "generated_at": _now_utc(),
        "schema_path": str(Path(args.schema)),
        "workspace_root": str(workspace_root),
        "bench_metrics_path": str(metrics_path) if metrics_path else None,
        "deterministic_scores": deterministic,
        "stage_scores": stage_scores,
        "judge_assessments": judge_assessments,
        "aggregate": aggregate,
        "artifacts_snapshot": {
            "entities_count": artifacts.get("entities_count"),
            "relationships_count": artifacts.get("relationships_count"),
            "community_reports_count": artifacts.get("community_reports_count"),
            "community_reports_placeholder": artifacts.get("community_reports_placeholder"),
            "expected_entity_hits": artifacts.get("expected_entity_hits"),
            "missing_files": artifacts.get("missing_files"),
            "query_output_chars": len((artifacts.get("query_output") or "")),
        },
    }
    validate_contract(report)
    json_path, md_path = write_reports(report, report_dir)

    print(f"Assessment JSON: {json_path}")
    print(f"Assessment Markdown: {md_path}")
    print(f"Final score: {aggregate['final_score']} (rank {aggregate['rank']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
