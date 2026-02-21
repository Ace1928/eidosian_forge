from __future__ import annotations

import json
import os
import time
from pathlib import Path

from code_forge.integration.provenance_registry import (
    build_provenance_registry,
    load_latest_benchmark_for_root,
    write_provenance_registry,
)


def test_build_provenance_registry_merges_unit_links() -> None:
    provenance = {
        "provenance_id": "prov_1",
        "generated_at": "2026-02-21T00:00:00+00:00",
        "stage": "archive_digester",
        "root_path": "/tmp/repo",
        "integration_policy": "effective_run",
        "integration_run_id": "run_123",
        "source_run_id": "run_123",
        "artifacts": [{"artifact_kind": "triage", "path": "/tmp/out/triage.json"}],
        "knowledge_links": {"count": 1, "links": [{"unit_id": "u1", "node_id": "n1", "status": "created"}]},
        "memory_links": {"count": 1, "links": [{"unit_id": "u1", "memory_id": "m1", "status": "created"}]},
        "graphrag_links": {
            "count": 1,
            "documents": [
                {
                    "unit_id": "u1",
                    "document_path": "/tmp/grag/u1.md",
                    "qualified_name": "pkg.mod.func",
                    "language": "python",
                    "unit_type": "function",
                    "source_file_path": "pkg/mod.py",
                }
            ],
        },
    }
    stage_summary = {
        "summary_path": "/tmp/out/archive_digester_summary.json",
        "ingestion_stats": {"files_processed": 4, "units_created": 3},
        "validation": {"pass": True},
    }
    drift = {
        "comparison": {
            "warnings": ["latency drift"],
            "comparisons": [{"metric": "search_p95", "delta": 12.0, "delta_pct": 10.0}],
        }
    }
    benchmark = {
        "generated_at": "2026-02-21T00:00:00+00:00",
        "ingestion": {"units_per_s": {"mean": 42.0}},
        "search": {"latency_ms": {"p95": 120.0}},
        "graph": {"build_ms": 800.0},
        "gate": {"pass": True, "violations": []},
    }

    registry = build_provenance_registry(
        provenance_payload=provenance,
        stage_summary_payload=stage_summary,
        drift_payload=drift,
        benchmark_payload=benchmark,
    )
    assert registry["schema_version"].startswith("code_forge_provenance_registry")
    unit_links = registry["links"]["unit_links"]
    assert len(unit_links) == 1
    assert unit_links[0]["unit_id"] == "u1"
    assert unit_links[0]["knowledge_node_id"] == "n1"
    assert unit_links[0]["memory_id"] == "m1"
    assert unit_links[0]["qualified_name"] == "pkg.mod.func"
    assert registry["benchmark"]["gate_pass"] is True
    assert registry["drift"]["warning_count"] == 1


def test_write_registry_and_load_latest_benchmark(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    reports = tmp_path / "reports"
    reports.mkdir()

    older = reports / "code_forge_benchmark_old.json"
    older.write_text(
        json.dumps({"config": {"root_path": str(root)}, "gate": {"pass": False}}),
        encoding="utf-8",
    )
    old_ts = time.time() - 5.0
    os.utime(older, (old_ts, old_ts))
    newer = reports / "code_forge_benchmark_new.json"
    newer.write_text(
        json.dumps({"config": {"root_path": str(root)}, "gate": {"pass": True}}),
        encoding="utf-8",
    )

    payload = load_latest_benchmark_for_root(root_path=root, search_roots=[reports])
    assert payload is not None
    assert payload["gate"]["pass"] is True

    registry = write_provenance_registry(
        output_path=out / "provenance_registry.json",
        provenance_payload={
            "provenance_id": "prov_2",
            "generated_at": "2026-02-21T00:00:00+00:00",
            "stage": "roundtrip",
            "root_path": str(root),
            "integration_policy": "global",
            "integration_run_id": None,
            "source_run_id": "run_2",
            "artifacts": [],
            "knowledge_links": {"count": 0, "links": []},
            "memory_links": {"count": 0, "links": []},
            "graphrag_links": {"count": 0, "documents": []},
        },
        benchmark_payload=payload,
    )
    assert Path(registry["path"]).exists()
