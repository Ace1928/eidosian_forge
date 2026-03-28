from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parents[1] / "export_entity_proof_bundle.py"
_SPEC = importlib.util.spec_from_file_location("export_entity_proof_bundle", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
export_bundle = _MODULE.export_bundle


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_export_bundle_collects_latest_proof_and_benchmarks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(
        repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json",
        json.dumps({"overall": {"status": "yellow", "score": 0.7}}),
    )
    _write(repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.md", "# proof\n")
    _write(
        repo_root / "reports" / "proof" / "migration_replay_scorecard_latest.json",
        json.dumps({"overall_score": 0.8, "status": "green"}),
    )
    _write(repo_root / "reports" / "proof" / "migration_replay_scorecard_latest.md", "# migration\n")
    _write(
        repo_root / "reports" / "proof" / "identity_continuity_scorecard_latest.json",
        json.dumps(
            {
                "overall_score": 0.77,
                "status": "yellow",
                "history": {"trend": "stable", "delta_from_previous": 0.01, "sample_count": 3},
            }
        ),
    )
    _write(repo_root / "reports" / "proof" / "identity_continuity_scorecard_latest.md", "# identity\n")
    _write(
        repo_root / "reports" / "proof" / "identity_continuity_scorecard_20260319_000000.json",
        json.dumps({"overall_score": 0.7, "status": "yellow", "generated_at": "2026-03-19T00:00:00Z"}),
    )
    _write(repo_root / "docs" / "THEORY_OF_OPERATION.md", "# theory\n")
    _write(
        repo_root / "reports" / "external_benchmarks" / "agencybench" / "latest.json",
        json.dumps({"suite": "agencybench", "score": 1.0, "status": "green"}),
    )
    _write(
        repo_root / "data" / "runtime" / "session_bridge" / "latest_context.json",
        json.dumps({"recent_sessions": [{"session_id": "codex:1"}]}),
    )
    _write(
        repo_root / "data" / "runtime" / "session_bridge" / "import_status.json",
        json.dumps({"last_sync_at": "2026-03-20T03:00:00Z", "gemini": {"imported_ids": ["g1"]}, "codex": {"threads": {"t1": 2}}}),
    )
    _write(
        repo_root / "doc_forge" / "runtime" / "processor_status.json",
        json.dumps({"status": "running", "phase": "processing"}),
    )
    _write(repo_root / "doc_forge" / "runtime" / "processor_history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "eidos_scheduler_status.json",
        json.dumps({"state": "sleeping", "current_task": "living_pipeline", "phase": "cycle_complete"}),
    )
    _write(repo_root / "data" / "runtime" / "eidos_scheduler_history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "qwenchat" / "status.json",
        json.dumps({"status": "running", "phase": "interactive"}),
    )
    _write(repo_root / "data" / "runtime" / "qwenchat" / "history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "living_pipeline_status.json",
        json.dumps({"status": "running", "phase": "graphrag"}),
    )
    _write(repo_root / "data" / "runtime" / "living_pipeline_history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "docs_upsert_batch_status.json",
        json.dumps({"status": "completed", "path_prefix": "doc_forge/src/doc_forge"}),
    )
    _write(repo_root / "data" / "runtime" / "docs_upsert_batch_history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "runtime_artifact_audit_status.json",
        json.dumps({"status": "completed", "tracked_violation_count": 3}),
    )
    _write(repo_root / "data" / "runtime" / "runtime_artifact_audit_history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "proof_refresh_status.json",
        json.dumps({"status": "completed"}),
    )
    _write(
        repo_root / "reports" / "word_forge_bridge_audit" / "latest.json",
        json.dumps({
            "bridge_counts": {"fully_bridged": 1, "partially_bridged": 3, "any_bridged": 4},
            "bridge_quality": {"candidate_term_count": 4, "fully_bridged_ratio": 0.25},
            "community_summary": {"community_count": 2},
        }),
    )
    _write(repo_root / "reports" / "word_forge_bridge_audit" / "latest.md", "# Word Forge Bridge\n")
    _write(
        repo_root / "data" / "runtime" / "word_forge_bridge_audit_status.json",
        json.dumps({"status": "completed", "phase": "completed"}),
    )
    _write(repo_root / "data" / "runtime" / "word_forge_bridge_audit_history.jsonl", "{}\n")
    _write(
        repo_root / "data" / "runtime" / "runtime_benchmark_run_status.json",
        json.dumps({"status": "completed", "scenario": "scenario2", "engine": "local_agent"}),
    )
    _write(
        repo_root / "reports" / "runtime_artifact_audit" / "latest.json",
        json.dumps({"tracked_violation_count": 3, "live_generated_count": 9}),
    )
    _write(repo_root / "reports" / "runtime_artifact_audit" / "latest.md", "# Runtime Artifact Audit\n")
    _write(
        repo_root / "data" / "runtime" / "external_benchmarks" / "agencybench" / "scenario2" / "20260320_010203" / "status.json",
        json.dumps(
            {
                "scenario": "scenario2",
                "engine": "local_agent",
                "model": "qwen3.5:2b",
                "status": "success",
                "stop_reason": "completed",
                "completed_count": 5,
                "attempt_count": 5,
                "generated_at": "2026-03-20T04:00:00Z",
            }
        ),
    )
    _write(
        repo_root / "data" / "runtime" / "external_benchmarks" / "agencybench" / "scenario2" / "20260320_010203" / "attempts.jsonl",
        '{"attempt": 1}\n',
    )

    result = export_bundle(repo_root, repo_root / "reports" / "proof_bundle")

    manifest_path = repo_root / result["manifest"]
    bundle_path = repo_root / result["bundle"]
    assert manifest_path.exists()
    assert bundle_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["proof_summary"]["status"] == "yellow"
    assert manifest["proof_summary"]["score"] == 0.7
    assert manifest["migration_summary"]["status"] == "green"
    assert manifest["identity_summary"]["history"]["trend"] == "stable"
    assert len(manifest["identity_summary"]["recent_history"]) == 1
    assert manifest["session_bridge_summary"]["imported_records"] == 2
    assert manifest["word_forge_bridge_summary"]["fully_bridged"] == 1
    assert manifest["word_forge_bridge_summary"]["community_count"] == 2
    assert manifest["operator_jobs_summary"]["proof_refresh"]["status"] == "completed"
    assert manifest["operator_jobs_summary"]["docs_batch"]["status"] == "completed"
    assert manifest["operator_jobs_summary"]["runtime_artifact_audit"]["tracked_violation_count"] == 3
    assert manifest["runtime_service_summary"]["scheduler_status"] == "sleeping"
    assert manifest["runtime_service_summary"]["scheduler_task"] == "living_pipeline"
    assert manifest["runtime_service_summary"]["scheduler_phase"] == "cycle_complete"
    assert manifest["runtime_service_summary"]["doc_processor_phase"] == "processing"
    assert manifest["runtime_service_summary"]["qwenchat_phase"] == "interactive"
    assert manifest["runtime_service_summary"]["living_pipeline_phase"] == "graphrag"
    assert manifest["runtime_service_summary"]["docs_batch_status"] == "completed"
    assert manifest["runtime_service_summary"]["runtime_artifact_audit_status"] == "completed"
    assert manifest["runtime_service_summary"]["runtime_artifact_audit_tracked_violations"] == 3
    assert manifest["runtime_service_summary"]["word_forge_bridge"]["status"] == "completed"
    assert manifest["benchmarks"][0]["suite"] == "agencybench"
    assert manifest["runtime_benchmarks"][0]["scenario"] == "scenario2"
    assert manifest["runtime_benchmarks"][0]["status"] == "success"
    assert manifest["missing"] == []
    assert any(item["label"] == "identity_continuity_json" for item in manifest["files"])
    assert any(item["label"].startswith("identity_history:") for item in manifest["files"])
    assert any(item["label"] == "session_bridge_context" for item in manifest["files"])
    assert any(item["label"] == "word_forge_bridge_report" for item in manifest["files"])
    assert any(item["label"].startswith("runtime_benchmark:") for item in manifest["files"])
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("manifest.json") for name in names)
    assert any(name.endswith("external_benchmarks/agencybench/latest.json") for name in names)
    assert any(name.endswith("runtime/session_bridge/latest_context.json") for name in names)
    assert any(name.endswith("runtime/scheduler/status.json") for name in names)
    assert any(name.endswith("runtime/doc_processor/status.json") for name in names)
    assert any(name.endswith("runtime/qwenchat/status.json") for name in names)
    assert any(name.endswith("runtime/living_pipeline_status.json") for name in names)
    assert any(name.endswith("runtime/docs_batch/status.json") for name in names)
    assert any(name.endswith("runtime/runtime_artifact_audit/status.json") for name in names)
    assert any(name.endswith("reports/word_forge_bridge_audit/latest.json") for name in names)
    assert any(name.endswith("reports/runtime_artifact_audit/latest.json") for name in names)
    assert any(name.endswith("runtime_benchmarks/agencybench/scenario2/20260320_010203/status.json") for name in names)


def test_export_bundle_reports_missing_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(
        repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json",
        json.dumps({"overall": {"status": "red", "score": 0.2}}),
    )

    result = export_bundle(repo_root, repo_root / "reports" / "proof_bundle")
    manifest = json.loads((repo_root / result["manifest"]).read_text(encoding="utf-8"))
    assert "migration_json" in manifest["missing"]
    assert "theory_of_operation" in manifest["missing"]
