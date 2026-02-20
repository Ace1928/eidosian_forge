from __future__ import annotations

import json
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.ingest.runner import IngestionRunner
from code_forge.library.db import CodeLibraryDB


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    idx = int((pct / 100.0) * (len(ordered) - 1))
    return float(ordered[max(0, min(idx, len(ordered) - 1))])


@dataclass
class BenchmarkConfig:
    root_path: str
    extensions: list[str]
    max_files: Optional[int]
    ingestion_repeats: int
    query_repeats: int
    queries: list[str]
    max_regression_pct: float


@dataclass
class BenchmarkResult:
    generated_at: str
    config: dict[str, Any]
    ingestion: dict[str, Any]
    search: dict[str, Any]
    graph: dict[str, Any]
    gate: dict[str, Any]


def _default_queries() -> list[str]:
    return [
        "consciousness kernel status",
        "workspace competition winner",
        "semantic search similarity",
        "knowledge graph integration",
        "benchmark regression gate",
    ]


def _run_ingestion_benchmark(
    root_path: Path,
    extensions: list[str],
    max_files: Optional[int],
    repeats: int,
) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for _ in range(max(1, repeats)):
        with tempfile.TemporaryDirectory(prefix="code_forge_bench_") as tmp:
            base = Path(tmp)
            db = CodeLibraryDB(base / "library.sqlite")
            runner = IngestionRunner(db=db, runs_dir=base / "runs")

            start = time.monotonic()
            stats = runner.ingest_path(
                root_path,
                mode="analysis",
                extensions=extensions,
                max_files=max_files,
                progress_every=0,
            )
            elapsed = max(time.monotonic() - start, 1e-9)
            runs.append(
                {
                    "elapsed_s": elapsed,
                    "files_processed": stats.files_processed,
                    "units_created": stats.units_created,
                    "files_per_s": stats.files_processed / elapsed,
                    "units_per_s": stats.units_created / elapsed,
                }
            )

    elapsed = [r["elapsed_s"] for r in runs]
    files_per_s = [r["files_per_s"] for r in runs]
    units_per_s = [r["units_per_s"] for r in runs]
    return {
        "runs": runs,
        "run_count": len(runs),
        "elapsed_s": {
            "mean": statistics.mean(elapsed) if elapsed else 0.0,
            "p95": _percentile(elapsed, 95),
        },
        "files_per_s": {
            "mean": statistics.mean(files_per_s) if files_per_s else 0.0,
            "p95": _percentile(files_per_s, 95),
        },
        "units_per_s": {
            "mean": statistics.mean(units_per_s) if units_per_s else 0.0,
            "p95": _percentile(units_per_s, 95),
        },
    }


def _run_search_benchmark(
    db: CodeLibraryDB,
    queries: Iterable[str],
    repeats: int,
) -> dict[str, Any]:
    latencies_ms: list[float] = []
    query_stats: list[dict[str, Any]] = []

    for query in queries:
        query = str(query).strip()
        if not query:
            continue
        match_counts: list[int] = []
        for _ in range(max(1, repeats)):
            start = time.monotonic()
            matches = db.semantic_search(query, limit=20, min_score=0.01)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            match_counts.append(len(matches))
        query_stats.append(
            {
                "query": query,
                "avg_matches": statistics.mean(match_counts) if match_counts else 0.0,
            }
        )

    return {
        "query_count": len(query_stats),
        "query_stats": query_stats,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms) if latencies_ms else 0.0,
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "max": max(latencies_ms) if latencies_ms else 0.0,
        },
    }


def _run_graph_benchmark(db: CodeLibraryDB) -> dict[str, Any]:
    start = time.monotonic()
    graph = db.module_dependency_graph(rel_types=["imports", "calls", "uses"], limit_edges=30000)
    elapsed_ms = (time.monotonic() - start) * 1000.0
    return {
        "build_ms": elapsed_ms,
        "summary": graph.get("summary", {}),
    }


def _run_regression_gate(
    baseline: Optional[dict[str, Any]],
    current: dict[str, Any],
    max_regression_pct: float,
) -> dict[str, Any]:
    if not baseline:
        return {
            "pass": True,
            "baseline_loaded": False,
            "violations": [],
            "max_regression_pct": max_regression_pct,
        }

    violations: list[str] = []
    loss_factor = 1.0 - max_regression_pct / 100.0
    gain_factor = 1.0 + max_regression_pct / 100.0

    base_units = float(((baseline.get("ingestion") or {}).get("units_per_s") or {}).get("mean") or 0.0)
    cur_units = float(((current.get("ingestion") or {}).get("units_per_s") or {}).get("mean") or 0.0)
    if base_units > 0 and cur_units < (base_units * loss_factor):
        violations.append(f"Ingestion units/s regressed: baseline={base_units:.4f}, current={cur_units:.4f}")

    base_search = float(((baseline.get("search") or {}).get("latency_ms") or {}).get("p95") or 0.0)
    cur_search = float(((current.get("search") or {}).get("latency_ms") or {}).get("p95") or 0.0)
    if base_search > 0 and cur_search > (base_search * gain_factor) and (cur_search - base_search) > 20.0:
        violations.append(f"Search latency p95 regressed: baseline={base_search:.4f}ms, current={cur_search:.4f}ms")

    base_graph = float(((baseline.get("graph") or {}).get("build_ms") or 0.0))
    cur_graph = float(((current.get("graph") or {}).get("build_ms") or 0.0))
    if base_graph > 0 and cur_graph > (base_graph * gain_factor) and (cur_graph - base_graph) > 50.0:
        violations.append(f"Dependency graph build regressed: baseline={base_graph:.4f}ms, current={cur_graph:.4f}ms")

    return {
        "pass": len(violations) == 0,
        "baseline_loaded": True,
        "violations": violations,
        "max_regression_pct": max_regression_pct,
    }


def run_benchmark_suite(
    *,
    root_path: Path,
    db_path: Path,
    runs_dir: Path,
    output_path: Path,
    extensions: Optional[Iterable[str]] = None,
    max_files: Optional[int] = None,
    ingestion_repeats: int = 1,
    query_repeats: int = 5,
    queries: Optional[Iterable[str]] = None,
    baseline_path: Optional[Path] = None,
    max_regression_pct: float = 25.0,
    prepare_ingest: bool = True,
    write_baseline: bool = False,
) -> dict[str, Any]:
    root_path = Path(root_path).resolve()
    db_path = Path(db_path).resolve()
    runs_dir = Path(runs_dir).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    extension_list = sorted({e.lower() for e in (extensions or GenericCodeAnalyzer.supported_extensions())})
    query_list = [q for q in (queries or _default_queries()) if str(q).strip()]

    config = BenchmarkConfig(
        root_path=str(root_path),
        extensions=extension_list,
        max_files=max_files,
        ingestion_repeats=max(1, int(ingestion_repeats)),
        query_repeats=max(1, int(query_repeats)),
        queries=query_list,
        max_regression_pct=float(max_regression_pct),
    )

    if prepare_ingest:
        db = CodeLibraryDB(db_path)
        runner = IngestionRunner(db=db, runs_dir=runs_dir)
        runner.ingest_path(
            root_path,
            mode="analysis",
            extensions=extension_list,
            max_files=max_files,
            progress_every=200,
        )

    ingestion = _run_ingestion_benchmark(
        root_path=root_path,
        extensions=extension_list,
        max_files=max_files,
        repeats=config.ingestion_repeats,
    )

    db = CodeLibraryDB(db_path)
    search = _run_search_benchmark(db=db, queries=query_list, repeats=config.query_repeats)
    graph = _run_graph_benchmark(db=db)

    current = {
        "ingestion": ingestion,
        "search": search,
        "graph": graph,
    }

    baseline_payload = None
    if baseline_path is not None and baseline_path.exists():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))

    gate = _run_regression_gate(
        baseline=baseline_payload,
        current=current,
        max_regression_pct=config.max_regression_pct,
    )

    result = BenchmarkResult(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        config=asdict(config),
        ingestion=ingestion,
        search=search,
        graph=graph,
        gate=gate,
    )

    payload = asdict(result)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if write_baseline and baseline_path is not None:
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return payload
