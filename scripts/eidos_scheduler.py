#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from eidosian_core import eidosian
from eidosian_runtime import ForgeRuntimeCoordinator

from living_knowledge_pipeline import (
    COORDINATOR_STATUS_PATH,
    FORGE_ROOT,
    SCHEDULER_STATUS_PATH,
    LivingDocumentationConfig,
    _now_utc,
    run_pipeline,
)


@eidosian()
def _write_scheduler_status(payload: dict[str, Any]) -> dict[str, Any]:
    SCHEDULER_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("contract", "eidos.scheduler_status.v1")
    payload["updated_at"] = _now_utc()
    SCHEDULER_STATUS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


@eidosian()
def _default_queries() -> list[str]:
    return [
        "What are the main active knowledge communities and their current risks?",
        "Summarize current code library health, duplication pressure, and graph coverage.",
        "What changed in memory, lexicon, and documentation since the previous run?",
    ]


@eidosian()
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Eidosian sequential scheduler for the living knowledge pipeline.")
    parser.add_argument("--repo-root", default=str(FORGE_ROOT))
    parser.add_argument("--output-root", default=str(FORGE_ROOT / "reports" / "living_knowledge"))
    parser.add_argument("--workspace-root", default=str(FORGE_ROOT / "data" / "living_knowledge" / "workspace"))
    parser.add_argument("--interval-sec", type=float, default=float(os.environ.get("EIDOS_SCHEDULER_INTERVAL_SEC", "1800")))
    parser.add_argument("--max-file-bytes", type=int, default=2_000_000)
    parser.add_argument("--max-chars-per-doc", type=int, default=20_000)
    parser.add_argument("--code-max-files", type=int, default=0)
    parser.add_argument("--method", default="native")
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--run-graphrag", action="store_true")
    parser.add_argument("--skip-graphrag", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--doc-model", default=os.environ.get("EIDOS_LIVING_DOC_MODEL", "qwen3.5:2b"))
    parser.add_argument("--doc-thinking-mode", default=os.environ.get("EIDOS_LIVING_DOC_THINKING_MODE", "on"))
    parser.add_argument("--doc-timeout-sec", type=float, default=float(os.environ.get("EIDOS_LIVING_DOC_TIMEOUT_SEC", "900")))
    parser.add_argument("--doc-max-tokens", type=int, default=int(os.environ.get("EIDOS_LIVING_DOC_MAX_TOKENS", "1400")))
    parser.add_argument("--doc-temperature", type=float, default=float(os.environ.get("EIDOS_LIVING_DOC_TEMPERATURE", "0.1")))
    return parser.parse_args()


@eidosian()
def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    workspace_root = Path(args.workspace_root).expanduser().resolve()
    queries = list(args.queries or _default_queries())
    run_graphrag = bool(args.run_graphrag) or not bool(args.skip_graphrag)
    code_max_files = None if int(args.code_max_files or 0) <= 0 else int(args.code_max_files)
    living_doc_config = LivingDocumentationConfig(
        enabled=True,
        model=str(args.doc_model),
        thinking_mode=str(args.doc_thinking_mode),
        timeout=float(args.doc_timeout_sec),
        max_tokens=int(args.doc_max_tokens),
        temperature=float(args.doc_temperature),
    )
    coordinator = ForgeRuntimeCoordinator(COORDINATOR_STATUS_PATH)

    cycle = 0
    consecutive_failures = 0
    while True:
        cycle += 1
        start_monotonic = time.monotonic()
        _write_scheduler_status(
            {
                "state": "running",
                "current_task": "living_pipeline",
                "cycle": cycle,
                "consecutive_failures": consecutive_failures,
                "interval_sec": float(args.interval_sec),
                "run_graphrag": run_graphrag,
                "doc_model": living_doc_config.model,
                "doc_thinking_mode": living_doc_config.thinking_mode,
                "repo_root": str(repo_root),
                "output_root": str(output_root),
                "workspace_root": str(workspace_root),
                "queries": queries,
                "status_path": str(SCHEDULER_STATUS_PATH),
            }
        )
        coordinator.queue_snapshot(
            jobs=[
                {
                    "id": "living_pipeline",
                    "kind": "pipeline",
                    "enabled": True,
                    "run_graphrag": run_graphrag,
                    "doc_model": living_doc_config.model,
                    "doc_thinking_mode": living_doc_config.thinking_mode,
                },
                {
                    "id": "word_forge",
                    "kind": "lexicon",
                    "enabled": True,
                    "depends_on": ["living_pipeline"],
                    "model": living_doc_config.model,
                    "thinking_mode": "on",
                },
                {
                    "id": "graphrag",
                    "kind": "knowledge_index",
                    "enabled": run_graphrag,
                    "depends_on": ["living_pipeline"],
                },
                {
                    "id": "living_documentation",
                    "kind": "report_generation",
                    "enabled": True,
                    "depends_on": ["living_pipeline"],
                    "model": living_doc_config.model,
                    "thinking_mode": living_doc_config.thinking_mode,
                },
                {
                    "id": "autonomy_supervisor",
                    "kind": "autonomy",
                    "enabled": True,
                },
                {
                    "id": "atlas_dashboard",
                    "kind": "ui",
                    "enabled": True,
                },
            ],
            policy={
                "max_active_model_families": 1,
                "max_active_model_instances": 2,
                "prefer_sequential_llm_tasks": True,
                "shared_embedding_contract": "eidos.embedding.default",
                "shared_vector_contract": "eidos.vector.hnsw",
                "doc_model": living_doc_config.model,
                "doc_thinking_mode": living_doc_config.thinking_mode,
            },
        )
        coordinator.heartbeat(
            owner="eidos_scheduler",
            task="living_pipeline",
            state="running",
            active_models=[],
            metadata={
                "cycle": cycle,
                "interval_sec": float(args.interval_sec),
                "doc_model": living_doc_config.model,
                "doc_thinking_mode": living_doc_config.thinking_mode,
            },
        )
        try:
            manifest = run_pipeline(
                repo_root=repo_root,
                output_root=output_root,
                workspace_root=workspace_root,
                max_file_bytes=int(args.max_file_bytes),
                max_chars_per_doc=int(args.max_chars_per_doc),
                code_max_files=code_max_files,
                run_graphrag=run_graphrag,
                queries=queries,
                method=str(args.method),
                living_doc_config=living_doc_config,
            )
            elapsed = round(max(0.0, time.monotonic() - start_monotonic), 3)
            consecutive_failures = 0
            _write_scheduler_status(
                {
                    "state": "idle",
                    "current_task": "sleep",
                    "cycle": cycle,
                    "consecutive_failures": 0,
                    "interval_sec": float(args.interval_sec),
                    "run_graphrag": run_graphrag,
                    "doc_model": living_doc_config.model,
                    "doc_thinking_mode": living_doc_config.thinking_mode,
                    "last_success_at": _now_utc(),
                    "last_elapsed_seconds": elapsed,
                    "next_run_in_seconds": float(args.interval_sec),
                    "last_run_id": manifest.get("run_id"),
                    "last_manifest_path": str(output_root / str(manifest.get("run_id")) / "manifest.json"),
                    "latest_pipeline_status_path": str((FORGE_ROOT / "data" / "runtime" / "living_pipeline_status.json")),
                    "summary": {
                        "records_total": manifest.get("records_total"),
                        "records_by_kind": manifest.get("records_by_kind"),
                        "word_forge": manifest.get("word_forge"),
                        "graphrag": {
                            "indexed": bool((manifest.get("graphrag") or {}).get("indexed")),
                            "assessment_summary": (manifest.get("graphrag") or {}).get("assessment_summary"),
                        },
                        "living_documentation": manifest.get("living_documentation"),
                    },
                }
            )
            coordinator.heartbeat(
                owner="eidos_scheduler",
                task="sleep",
                state="idle",
                active_models=[],
                metadata={
                    "cycle": cycle,
                    "last_run_id": manifest.get("run_id"),
                    "next_run_in_seconds": float(args.interval_sec),
                    "records_total": manifest.get("records_total"),
                    "summary": {
                        "word_forge_updated": bool((manifest.get("word_forge") or {}).get("updated")),
                        "graphrag_indexed": bool((manifest.get("graphrag") or {}).get("indexed")),
                    },
                },
            )
        except Exception as exc:
            consecutive_failures += 1
            _write_scheduler_status(
                {
                    "state": "error",
                    "current_task": "living_pipeline",
                    "cycle": cycle,
                    "consecutive_failures": consecutive_failures,
                    "interval_sec": float(args.interval_sec),
                    "run_graphrag": run_graphrag,
                    "doc_model": living_doc_config.model,
                    "doc_thinking_mode": living_doc_config.thinking_mode,
                    "last_error": str(exc),
                    "last_error_trace": traceback.format_exc(limit=12),
                    "next_run_in_seconds": float(args.interval_sec),
                    "latest_pipeline_status_path": str((FORGE_ROOT / "data" / "runtime" / "living_pipeline_status.json")),
                }
            )
            coordinator.heartbeat(
                owner="eidos_scheduler",
                task="living_pipeline",
                state="error",
                active_models=[],
                metadata={
                    "cycle": cycle,
                    "consecutive_failures": consecutive_failures,
                    "last_error": str(exc),
                },
            )
            if args.once:
                return 1

        if args.once:
            return 0
        time.sleep(max(5.0, float(args.interval_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
