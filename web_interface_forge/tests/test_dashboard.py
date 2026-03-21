from __future__ import annotations

import json
import subprocess
import sys
import time
import types
from pathlib import Path

from fastapi.testclient import TestClient
from file_forge import FileForge
from web_interface_forge.dashboard import main as dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_doc_status_api_and_index_page(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "doc_forge" / "runtime"
    final_docs = runtime / "final_docs"
    final_docs.mkdir(parents=True, exist_ok=True)

    _write_json(
        runtime / "processor_status.json",
        {
            "contract": "eidos.doc_processor.status.v1",
            "status": "running",
            "phase": "processing",
            "processed": 12,
            "remaining": 4,
            "average_quality_score": 0.88,
            "last_approved": "foo/bar.py",
        },
    )
    (runtime / "processor_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.doc_processor.status.v1",
                "status": "running",
                "phase": "processing",
                "processed": 12,
                "generated_at": "2026-03-20T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime / "doc_index.json",
        {
            "entries": [
                {
                    "source": "foo/bar.py",
                    "document": "foo/bar.py.md",
                    "score": 0.91,
                    "doc_type": "py",
                    "updated_at": "2026-02-26T00:00:00+00:00",
                }
            ]
        },
    )

    (final_docs / "foo").mkdir(parents=True, exist_ok=True)
    (final_docs / "foo" / "bar.py.md").write_text("# Example\n", encoding="utf-8")
    file_forge_db = tmp_path / "data" / "file_forge" / "library.sqlite"
    FileForge(base_path=tmp_path).index_directory(runtime, db_path=file_forge_db)
    runtime_dir = tmp_path / "data" / "runtime"
    _write_json(
        runtime_dir / "local_mcp_agent" / "status.json",
        {
            "status": "success",
            "profile": "observer",
            "tool_calls": 1,
            "mcp_transport": "stdio",
            "created_at": "2026-03-07T00:00:00+00:00",
        },
    )
    _write_json(
        runtime_dir / "qwenchat" / "status.json",
        {
            "contract": "eidos.qwenchat.status.v1",
            "status": "running",
            "phase": "interactive",
            "session_id": "qwenchat:test",
            "model": "qwen3.5:2b",
        },
    )
    (runtime_dir / "qwenchat" / "history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.qwenchat.status.v1",
                "status": "running",
                "phase": "interactive",
                "session_id": "qwenchat:test",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "local_mcp_agent" / "history.jsonl").write_text(
        json.dumps(
            {
                "status": "success",
                "tool_calls": 1,
                "mcp_transport": "stdio",
                "created_at": "2026-03-07T00:00:00+00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime_dir / "living_pipeline_status.json",
        {
            "contract": "eidos.living_pipeline.status.v1",
            "status": "running",
            "phase": "graphrag",
            "run_id": "20260320_010203",
            "records_total": 42,
        },
    )
    (runtime_dir / "living_pipeline_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.living_pipeline.status.v1",
                "status": "running",
                "phase": "graphrag",
                "run_id": "20260320_010203",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime_dir / "eidos_scheduler_status.json",
        {
            "state": "sleeping",
            "current_task": "living_pipeline",
            "phase": "cycle_complete",
        },
    )
    (runtime_dir / "eidos_scheduler_history.jsonl").write_text(
        json.dumps(
            {
                "state": "sleeping",
                "current_task": "living_pipeline",
                "cycle": 2,
                "updated_at": "2026-03-20T00:20:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime_dir / "forge_coordinator_status.json",
        {
            "state": "idle",
            "task": "idle",
            "owner": "",
            "active_models": [],
        },
    )
    _write_json(
        runtime_dir / "directory_docs_status.json",
        {
            "coverage_ratio": 0.995,
            "missing_readme_count": 2,
            "required_directory_count": 400,
            "coverage_delta": 0.005,
            "missing_delta": -1,
            "drift_state": "improved",
            "missing_examples": ["doc_forge/src/doc_forge/scribe"],
        },
    )
    _write_json(
        runtime_dir / "directory_docs_history.json",
        {
            "entries": [
                {
                    "coverage_ratio": 0.99,
                    "missing_readme_count": 3,
                    "missing_delta": 0,
                    "drift_state": "stable",
                },
                {
                    "coverage_ratio": 0.995,
                    "missing_readme_count": 2,
                    "missing_delta": -1,
                    "drift_state": "improved",
                },
            ]
        },
    )
    _write_json(
        runtime_dir / "session_bridge" / "latest_context.json",
        {
            "contract": "eidos.session_context.v1",
            "session_id": "qwenchat:test",
            "recent_sessions": [{"session_id": "codex:abc", "interface": "codex", "events": [{"summary": "recent codex"}]}],
        },
    )
    _write_json(
        runtime_dir / "session_bridge" / "import_status.json",
        {
            "gemini": {"imported_ids": ["a1"]},
            "codex": {"threads": {"t1": 1}},
            "last_sync_at": "2026-03-20T00:00:00+00:00",
        },
    )
    _write_json(
        runtime_dir / "external_benchmarks" / "agencybench" / "scenario2" / "20260320_000000" / "status.json",
        {
            "contract": "eidos.agencybench_runtime_status.v1",
            "scenario": "scenario2",
            "engine": "local_agent",
            "model": "qwen3.5:2b",
            "status": "running",
            "stop_reason": "subtask1_attempt_1",
            "completed_count": 1,
            "attempt_count": 1,
            "generated_at": "2026-03-20T00:10:00Z",
            "run_root": str(runtime_dir / "external_benchmarks" / "agencybench" / "scenario2" / "20260320_000000"),
        },
    )
    _write_json(
        tmp_path / "reports" / "proof_bundle" / "latest_manifest.json",
        {
            "contract": "eidos.entity_proof_bundle.v1",
            "bundle_root": "20260320_033604",
            "benchmarks": [{"suite": "agencybench"}],
            "missing": [],
            "session_bridge_summary": {"imported_records": 3},
        },
    )
    _write_json(
        runtime_dir / "proof_refresh_status.json",
        {
            "contract": "eidos.proof_refresh.status.v1",
            "status": "completed",
            "window_days": 30,
            "proof_returncode": 0,
            "bundle_returncode": 0,
            "started_at": "2026-03-20T00:30:00Z",
        },
    )
    (runtime_dir / "proof_refresh_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.proof_refresh.status.v1",
                "status": "completed",
                "window_days": 30,
                "proof_returncode": 0,
                "finished_at": "2026-03-20T00:31:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime_dir / "runtime_benchmark_run_status.json",
        {
            "contract": "eidos.runtime_benchmark_run.status.v1",
            "status": "completed",
            "scenario": "scenario2",
            "engine": "local_agent",
            "returncode": 0,
            "finished_at": "2026-03-20T00:40:00Z",
        },
    )
    (runtime_dir / "runtime_benchmark_run_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.runtime_benchmark_run.status.v1",
                "status": "completed",
                "scenario": "scenario2",
                "engine": "local_agent",
                "finished_at": "2026-03-20T00:41:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime_dir / "docs_upsert_batch_status.json",
        {
            "contract": "eidos.docs_upsert_batch.status.v1",
            "status": "completed",
            "limit": 5,
            "path_prefix": "doc_forge/src/doc_forge",
            "dry_run": True,
            "finished_at": "2026-03-20T00:42:00Z",
        },
    )
    (runtime_dir / "docs_upsert_batch_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.docs_upsert_batch.status.v1",
                "status": "completed",
                "limit": 5,
                "path_prefix": "doc_forge/src/doc_forge",
                "finished_at": "2026-03-20T00:42:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        runtime_dir / "runtime_artifact_audit_status.json",
        {
            "contract": "eidos.runtime_artifact_audit.status.v1",
            "status": "completed",
            "tracked_violation_count": 3,
            "live_generated_count": 9,
            "latest_report": "reports/runtime_artifact_audit/runtime_artifact_audit_20260320_004500.json",
            "finished_at": "2026-03-20T00:45:00Z",
        },
    )
    _write_json(
        runtime_dir / "code_forge_provenance_audit_status.json",
        {
            "contract": "eidos.code_forge_provenance_audit.status.v1",
            "status": "completed",
            "link_file_count": 0,
            "registry_file_count": 0,
            "invalid_file_count": 0,
            "latest_report": "reports/code_forge_provenance_audit/latest.json",
            "finished_at": "2026-03-20T00:46:00Z",
        },
    )
    (runtime_dir / "runtime_artifact_audit_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.runtime_artifact_audit.status.v1",
                "status": "completed",
                "tracked_violation_count": 3,
                "live_generated_count": 9,
                "finished_at": "2026-03-20T00:45:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (runtime_dir / "code_forge_provenance_audit_history.jsonl").write_text(
        json.dumps(
            {
                "contract": "eidos.code_forge_provenance_audit.status.v1",
                "status": "completed",
                "link_file_count": 0,
                "registry_file_count": 0,
                "finished_at": "2026-03-20T00:46:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "reports" / "runtime_artifact_audit" / "latest.json",
        {
            "repo_root": str(tmp_path),
            "tracked_violation_count": 3,
            "live_generated_count": 9,
        },
    )
    _write_json(
        tmp_path / "reports" / "code_forge_provenance_audit" / "latest.json",
        {
            "contract": "eidos.code_forge_provenance_audit.v1",
            "link_file_count": 0,
            "registry_file_count": 0,
            "invalid_file_count": 0,
        },
    )
    (tmp_path / "reports" / "runtime_artifact_audit" / "latest.md").write_text(
        "# Runtime Artifact Audit\n",
        encoding="utf-8",
    )
    (tmp_path / "reports" / "code_forge_provenance_audit" / "latest.md").write_text(
        "# Code Forge Provenance Audit\n",
        encoding="utf-8",
    )
    _write_json(
        tmp_path / "reports" / "proof" / "entity_proof_scorecard_latest.json",
        {
            "contract": "eidos.entity_proof_scorecard.v1",
            "overall": {"score": 0.74, "status": "yellow"},
            "freshness": {"status": "yellow"},
            "regression": {"status": "stable"},
            "categories": [{"category": "external_validity", "status": "yellow", "score": 0.7}],
        },
    )
    _write_json(
        tmp_path / "reports" / "proof" / "entity_proof_scorecard_20260320_000000.json",
        {
            "contract": "eidos.entity_proof_scorecard.v1",
            "generated_at": "2026-03-20T00:00:00Z",
            "overall": {"score": 0.71, "status": "yellow"},
            "freshness": {"status": "yellow"},
            "regression": {"status": "stable"},
        },
    )
    _write_json(
        tmp_path / "reports" / "proof" / "identity_continuity_scorecard_latest.json",
        {
            "contract": "eidos.identity_continuity_scorecard.v1",
            "overall_score": 0.93,
            "status": "green",
            "history": {"trend": "improved", "delta_from_previous": 0.05, "sample_count": 3},
            "session_bridge": {"recent_sessions": 2},
        },
    )
    _write_json(
        tmp_path / "reports" / "proof" / "identity_continuity_scorecard_20260320_010101.json",
        {
            "contract": "eidos.identity_continuity_scorecard.v1",
            "generated_at": "2026-03-20T01:01:01Z",
            "overall_score": 0.88,
            "status": "yellow",
            "session_bridge": {"recent_sessions": 1},
        },
    )
    _write_json(
        tmp_path / "reports" / "proof" / "identity_continuity_scorecard_20260320_020202.json",
        {
            "contract": "eidos.identity_continuity_scorecard.v1",
            "generated_at": "2026-03-20T02:02:02Z",
            "overall_score": 0.93,
            "status": "green",
            "session_bridge": {"recent_sessions": 2},
        },
    )
    _write_json(
        tmp_path / "reports" / "security" / "dependabot_open_summary_2026-03-20.json",
        {
            "totals": {"alerts": 15, "open": 15, "fixed": 0},
            "open_by_severity": {"critical": 1, "high": 10, "medium": 3, "low": 1},
            "top_packages": [["nltk", 3], ["authlib", 3], ["PyJWT", 2]],
        },
    )
    _write_json(
        tmp_path / "reports" / "security" / "dependabot_remediation_plan_2026-03-20.json",
        {
            "repo": "Ace1928/eidosian_forge",
            "batches": [{"name": "batch-1", "alerts": [232, 235, 243]}],
        },
    )
    target = tmp_path / "doc_forge" / "src" / "doc_forge" / "scribe"
    target.mkdir(parents=True, exist_ok=True)
    (target / "service.py").write_text(
        'from fastapi import FastAPI\napp = FastAPI()\n@app.get("/health")\ndef health():\n    return {"ok": True}\n',
        encoding="utf-8",
    )
    (tmp_path / "cfg").mkdir(parents=True, exist_ok=True)
    (tmp_path / "cfg" / "documentation_policy.json").write_text(
        '{"documented_prefixes":["doc_forge"],"excluded_prefixes":[],"excluded_segments":[]}',
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True, capture_output=True, text=True
    )
    subprocess.run(["git", "add", "."], cwd=str(tmp_path), check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(tmp_path), check=True, capture_output=True, text=True)

    monkeypatch.setattr(dashboard, "DOC_RUNTIME", runtime)
    monkeypatch.setattr(dashboard, "DOC_FINAL", final_docs)
    monkeypatch.setattr(dashboard, "DOC_INDEX", runtime / "doc_index.json")
    monkeypatch.setattr(dashboard, "DOC_STATUS", runtime / "processor_status.json")
    monkeypatch.setattr(dashboard, "DOC_HISTORY", runtime / "processor_history.jsonl")
    monkeypatch.setattr(dashboard, "FORGE_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "HOME_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(dashboard, "FILE_FORGE_DB", file_forge_db)
    monkeypatch.setattr(dashboard, "FILE_FORGE_INDEX_STATUS", runtime_dir / "file_forge_index_status.json")
    monkeypatch.setattr(dashboard, "FILE_FORGE_INDEX_HISTORY", runtime_dir / "file_forge_index_history.jsonl")
    monkeypatch.setattr(dashboard, "PROOF_REPORT_DIR", tmp_path / "reports" / "proof")
    monkeypatch.setattr(dashboard, "PROOF_BUNDLE_DIR", tmp_path / "reports" / "proof_bundle")
    monkeypatch.setattr(dashboard, "SECURITY_REPORT_DIR", tmp_path / "reports" / "security")
    monkeypatch.setattr(dashboard, "LOCAL_AGENT_STATUS", runtime_dir / "local_mcp_agent" / "status.json")
    monkeypatch.setattr(dashboard, "LOCAL_AGENT_HISTORY", runtime_dir / "local_mcp_agent" / "history.jsonl")
    monkeypatch.setattr(dashboard, "QWENCHAT_STATUS", runtime_dir / "qwenchat" / "status.json")
    monkeypatch.setattr(dashboard, "QWENCHAT_HISTORY", runtime_dir / "qwenchat" / "history.jsonl")
    monkeypatch.setattr(dashboard, "SCHEDULER_STATUS", runtime_dir / "eidos_scheduler_status.json")
    monkeypatch.setattr(dashboard, "SCHEDULER_HISTORY", runtime_dir / "eidos_scheduler_history.jsonl")
    monkeypatch.setattr(dashboard, "LIVING_PIPELINE_STATUS", runtime_dir / "living_pipeline_status.json")
    monkeypatch.setattr(dashboard, "LIVING_PIPELINE_HISTORY", runtime_dir / "living_pipeline_history.jsonl")
    monkeypatch.setattr(dashboard, "COORDINATOR_STATUS", runtime_dir / "forge_coordinator_status.json")
    monkeypatch.setattr(dashboard, "COORDINATOR_HISTORY", runtime_dir / "forge_runtime_trends.json")
    monkeypatch.setattr(dashboard, "DIRECTORY_DOCS_STATUS", runtime_dir / "directory_docs_status.json")
    monkeypatch.setattr(dashboard, "DIRECTORY_DOCS_HISTORY", runtime_dir / "directory_docs_history.json")
    monkeypatch.setattr(dashboard, "DOCS_BATCH_STATUS", runtime_dir / "docs_upsert_batch_status.json")
    monkeypatch.setattr(dashboard, "DOCS_BATCH_HISTORY", runtime_dir / "docs_upsert_batch_history.jsonl")
    monkeypatch.setattr(dashboard, "PROOF_REFRESH_STATUS", runtime_dir / "proof_refresh_status.json")
    monkeypatch.setattr(dashboard, "PROOF_REFRESH_HISTORY", runtime_dir / "proof_refresh_history.jsonl")
    monkeypatch.setattr(dashboard, "RUNTIME_BENCHMARK_RUN_STATUS", runtime_dir / "runtime_benchmark_run_status.json")
    monkeypatch.setattr(dashboard, "RUNTIME_BENCHMARK_RUN_HISTORY", runtime_dir / "runtime_benchmark_run_history.jsonl")
    monkeypatch.setattr(dashboard, "RUNTIME_ARTIFACT_AUDIT_STATUS", runtime_dir / "runtime_artifact_audit_status.json")
    monkeypatch.setattr(dashboard, "RUNTIME_ARTIFACT_AUDIT_HISTORY", runtime_dir / "runtime_artifact_audit_history.jsonl")
    monkeypatch.setattr(dashboard, "CODE_FORGE_PROVENANCE_AUDIT_STATUS", runtime_dir / "code_forge_provenance_audit_status.json")
    monkeypatch.setattr(dashboard, "CODE_FORGE_PROVENANCE_AUDIT_HISTORY", runtime_dir / "code_forge_provenance_audit_history.jsonl")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_PLAN_STATUS", runtime_dir / "code_forge_archive_plan_status.json")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_PLAN_HISTORY", runtime_dir / "code_forge_archive_plan_history.jsonl")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS", runtime_dir / "code_forge_archive_lifecycle_status.json")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY", runtime_dir / "code_forge_archive_lifecycle_history.jsonl")
    monkeypatch.setattr(dashboard, "SESSION_BRIDGE_CONTEXT", runtime_dir / "session_bridge" / "latest_context.json")
    monkeypatch.setattr(dashboard, "SESSION_BRIDGE_IMPORT_STATUS", runtime_dir / "session_bridge" / "import_status.json")
    monkeypatch.setattr(dashboard, "RUNTIME_ARTIFACT_REPORT_DIR", tmp_path / "reports" / "runtime_artifact_audit")
    monkeypatch.setattr(dashboard, "CODE_FORGE_PROVENANCE_REPORT_DIR", tmp_path / "reports" / "code_forge_provenance_audit")
    service_script = tmp_path / "eidos_termux_services.sh"
    service_script.write_text(
        "#!/bin/sh\n" "printf 'Atlas: runit run: /tmp/service: (pid 1) 10s; run: log: (pid 2) 10s\\n'\n",
        encoding="utf-8",
    )
    service_script.chmod(0o755)
    monkeypatch.setattr(dashboard, "SERVICES_SCRIPT", service_script)

    with TestClient(dashboard.app) as client:
        resp = client.get("/api/doc/status")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["index_count"] == 1
        assert payload["status"]["processed"] == 12

        html = client.get("/").text
        assert "foo/bar.py" in html
        assert "Indexed Docs" in html
        assert "Local Agent" in html
        assert "Doc Processor" in html
        assert "Qwenchat" in html
        assert "Living Pipeline" in html
        assert "Scheduler History" in html
        assert "Runtime Services" in html
        assert "Docs Batch" in html
        assert "Proof Refresh" in html
        assert "Benchmark Run" in html
        assert "Runtime Artifact Audit" in html
        assert "Code Forge Provenance Audit" in html
        assert "Docs Batch History" in html
        assert "Proof Refresh History" in html
        assert "Benchmark Run History" in html
        assert "Runtime Artifact Audit History" in html
        assert "Code Forge Provenance Audit History" in html
        assert "Doc Processor History" in html
        assert "Qwenchat History" in html
        assert "Operator Shell" in html
        assert "File Forge Operator" in html
        assert "Living Pipeline History" in html
        runtime_resp = client.get("/api/runtime")
        assert runtime_resp.status_code == 200
        runtime_payload = runtime_resp.json()
        assert runtime_payload["file_forge"]["total_files"] >= 1
        assert runtime_payload["local_agent"]["status"] == "success"
        assert runtime_payload["doc_processor"]["phase"] == "processing"
        assert runtime_payload["qwenchat"]["phase"] == "interactive"
        assert runtime_payload["living_pipeline"]["phase"] == "graphrag"
        assert runtime_payload["directory_docs"]["missing_readme_count"] == 2
        assert runtime_payload["session_bridge"]["context"]["session_id"] == "qwenchat:test"
        assert runtime_payload["proof_bundle"]["bundle_root"] == "20260320_033604"
        assert runtime_payload["proof"]["overall"]["score"] == 0.74
        assert runtime_payload["identity_continuity"]["overall_score"] == 0.93
        assert runtime_payload["identity_continuity"]["history"]["trend"] == "improved"
        assert len(runtime_payload["identity_history"]) == 2
        assert len(runtime_payload["proof_history"]) == 1
        assert runtime_payload["external_benchmarks"] == []
        assert runtime_payload["runtime_benchmarks"][0]["scenario"] == "scenario2"
        assert runtime_payload["proof_refresh"]["status"] == "completed"
        assert runtime_payload["proof_refresh_history"][0]["status"] == "completed"
        assert runtime_payload["runtime_benchmark_run"]["status"] == "completed"
        assert runtime_payload["runtime_benchmark_run_history"][0]["engine"] == "local_agent"
        assert runtime_payload["docs_batch"]["status"] == "completed"
        assert runtime_payload["docs_batch_history"][0]["path_prefix"] == "doc_forge/src/doc_forge"
        assert runtime_payload["runtime_artifact_audit"]["tracked_violation_count"] == 3
        assert runtime_payload["runtime_artifact_audit_history"][0]["live_generated_count"] == 9
        assert runtime_payload["security"]["totals"]["open"] == 15
        assert runtime_payload["security_plan"]["batches"][0]["name"] == "batch-1"
        runtime_services_resp = client.get("/api/runtime/services")
        assert runtime_services_resp.status_code == 200
        services = runtime_services_resp.json()["entries"]
        assert any(row["service"] == "scheduler" and row["phase"] == "cycle_complete" for row in services)
        assert any(row["service"] == "doc_processor" and row["phase"] == "processing" for row in services)
        assert any(row["service"] == "file_forge" for row in services)
        assert any(row["service"] == "qwenchat" and row["phase"] == "interactive" for row in services)
        assert any(row["service"] == "living_pipeline" and row["phase"] == "graphrag" for row in services)
        scheduler_resp = client.get("/api/runtime/scheduler")
        assert scheduler_resp.status_code == 200
        assert scheduler_resp.json()["status"]["state"] == "sleeping"
        assert scheduler_resp.json()["history"][0]["cycle"] == 2
        proof_refresh_resp = client.get("/api/proof/refresh/status")
        assert proof_refresh_resp.status_code == 200
        assert proof_refresh_resp.json()["status"] == "completed"
        proof_refresh_history_resp = client.get("/api/proof/refresh/history")
        assert proof_refresh_history_resp.status_code == 200
        assert proof_refresh_history_resp.json()["entries"][0]["proof_returncode"] == 0
        benchmark_run_resp = client.get("/api/benchmarks/runtime/run/status")
        assert benchmark_run_resp.status_code == 200
        assert benchmark_run_resp.json()["scenario"] == "scenario2"
        benchmark_run_history_resp = client.get("/api/benchmarks/runtime/run/history")
        assert benchmark_run_history_resp.status_code == 200
        assert benchmark_run_history_resp.json()["entries"][0]["engine"] == "local_agent"
        docs_batch_status_resp = client.get("/api/docs/upsert-batch/status")
        assert docs_batch_status_resp.status_code == 200
        assert docs_batch_status_resp.json()["status"] == "completed"
        docs_batch_history_resp = client.get("/api/docs/upsert-batch/history")
        assert docs_batch_history_resp.status_code == 200
        assert docs_batch_history_resp.json()["entries"][0]["limit"] == 5
        runtime_audit_status_resp = client.get("/api/runtime-artifacts/audit/status")
        assert runtime_audit_status_resp.status_code == 200
        assert runtime_audit_status_resp.json()["tracked_violation_count"] == 3
        runtime_audit_history_resp = client.get("/api/runtime-artifacts/audit/history")
        assert runtime_audit_history_resp.status_code == 200
        assert runtime_audit_history_resp.json()["entries"][0]["live_generated_count"] == 9
        local_agent_resp = client.get("/api/runtime/local-agent")
        assert local_agent_resp.status_code == 200
        assert local_agent_resp.json()["status"]["profile"] == "observer"
        doc_processor_resp = client.get("/api/runtime/doc-processor")
        assert doc_processor_resp.status_code == 200
        assert doc_processor_resp.json()["status"]["phase"] == "processing"
        assert doc_processor_resp.json()["history"][0]["processed"] == 12
        qwenchat_resp = client.get("/api/runtime/qwenchat")
        assert qwenchat_resp.status_code == 200
        assert qwenchat_resp.json()["status"]["session_id"] == "qwenchat:test"
        assert qwenchat_resp.json()["history"][0]["phase"] == "interactive"
        living_resp = client.get("/api/runtime/living-pipeline")
        assert living_resp.status_code == 200
        assert living_resp.json()["status"]["run_id"] == "20260320_010203"
        assert living_resp.json()["history"][0]["phase"] == "graphrag"
        docs_resp = client.get("/api/docs/coverage")
        assert docs_resp.status_code == 200
        assert docs_resp.json()["missing_readme_count"] == 2
        docs_tree_resp = client.get("/api/docs/tree")
        assert docs_tree_resp.status_code == 200
        assert docs_tree_resp.json()["contract"] == "eidos.documentation_tree.v1"
        docs_history_resp = client.get("/api/docs/history")
        assert docs_history_resp.status_code == 200
        assert len(docs_history_resp.json()["entries"]) == 2
        proof_bundle_resp = client.get("/api/proof/bundle/latest")
        assert proof_bundle_resp.status_code == 200
        assert proof_bundle_resp.json()["bundle_root"] == "20260320_033604"
        proof_summary_resp = client.get("/api/proof/summary")
        assert proof_summary_resp.status_code == 200
        assert proof_summary_resp.json()["proof"]["overall"]["score"] == 0.74
        monkeypatch.setattr(
            dashboard,
            "_run_proof_refresh_job",
            lambda *, window_days: _write_json(
                runtime_dir / "proof_refresh_status.json",
                {"contract": "eidos.proof_refresh.status.v1", "status": "completed", "window_days": window_days},
            ),
        )
        monkeypatch.setattr(
            dashboard,
            "_run_runtime_benchmark_job",
            lambda **kwargs: _write_json(
                runtime_dir / "runtime_benchmark_run_status.json",
                {"contract": "eidos.runtime_benchmark_run.status.v1", "status": "completed", **kwargs},
            ),
        )
        monkeypatch.setattr(
            dashboard,
            "_run_runtime_artifact_audit_job",
            lambda **kwargs: _write_json(
                runtime_dir / "runtime_artifact_audit_status.json",
                {"contract": "eidos.runtime_artifact_audit.status.v1", "status": "completed", **kwargs},
            ),
        )
        proof_refresh_run = client.post("/api/proof/refresh?background=false&window_days=14")
        assert proof_refresh_run.status_code == 200
        assert proof_refresh_run.json()["window_days"] == 14
        benchmark_run = client.post(
            "/api/benchmarks/runtime/run?background=false&scenario=scenario2&engine=local_agent&attempts_per_step=1&timeout_sec=900&keep_alive=4h"
        )
        assert benchmark_run.status_code == 200
        assert benchmark_run.json()["scenario"] == "scenario2"
        runtime_audit_run = client.post("/api/runtime-artifacts/audit?background=false")
        assert runtime_audit_run.status_code == 200
        assert runtime_audit_run.json()["status"] == "completed"
        proof_history_resp = client.get("/api/proof/history")
        assert proof_history_resp.status_code == 200
        assert len(proof_history_resp.json()["entries"]) == 1
        identity_resp = client.get("/api/proof/identity/latest")
        assert identity_resp.status_code == 200
        assert identity_resp.json()["overall_score"] == 0.93
        assert identity_resp.json()["history"]["trend"] == "improved"
        identity_history_resp = client.get("/api/proof/identity/history")
        assert identity_history_resp.status_code == 200
        assert len(identity_history_resp.json()["entries"]) == 2
        assert identity_history_resp.json()["entries"][-1]["overall_score"] == 0.93
        external_resp = client.get("/api/proof/external")
        assert external_resp.status_code == 200
        assert external_resp.json()["entries"] == []
        runtime_bench_resp = client.get("/api/benchmarks/runtime")
        assert runtime_bench_resp.status_code == 200
        assert runtime_bench_resp.json()["entries"][0]["engine"] == "local_agent"
        security_resp = client.get("/api/security/dependabot")
        assert security_resp.status_code == 200
        assert security_resp.json()["summary"]["totals"]["open"] == 15
        assert security_resp.json()["plan"]["batches"][0]["alerts"] == [232, 235, 243]
        session_bridge_resp = client.get("/api/session-bridge")
        assert session_bridge_resp.status_code == 200
        assert session_bridge_resp.json()["context"]["session_id"] == "qwenchat:test"
        render_resp = client.get("/api/docs/render", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert render_resp.status_code == 200
        assert "GET /health" in render_resp.json()["content"]
        upsert_resp = client.post("/api/docs/upsert", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert upsert_resp.status_code == 200
        batch_resp = client.post("/api/docs/upsert-batch", params={"limit": 5, "dry_run": True})
        assert batch_resp.status_code == 200
        assert batch_resp.json()["contract"] == "eidos.docs_upsert_batch.v2"
        queued_resp = client.post(
            "/api/docs/upsert-batch",
            params={"limit": 5, "dry_run": True, "background": True, "path_prefix": "doc_forge/src/doc_forge"},
        )
        assert queued_resp.status_code == 200
        assert queued_resp.json()["status"] == "queued"
        batch_status = client.get("/api/docs/upsert-batch/status")
        assert batch_status.status_code == 200
        readme_resp = client.get("/api/docs/readme", params={"path": "doc_forge/src/doc_forge/scribe"})
        assert readme_resp.status_code == 200
        assert "GET /health" in readme_resp.json()["content"]
        assert client.get("/browse/forge/").status_code == 200
        assert client.get("/browse/home/").status_code == 200


def test_browse_blocks_path_traversal() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.get("/browse/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403
        resp = client.get("/browse/forge/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403


def test_health_endpoint() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"


def test_service_api_invalid_action() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/services/invalid")
        assert resp.status_code == 400


def test_services_api_parses_status(monkeypatch, tmp_path: Path) -> None:
    service_script = tmp_path / "eidos_termux_services.sh"
    service_script.write_text(
        "#!/bin/sh\n"
        "printf 'Eidos Scheduler: runit run: /tmp/service: (pid 1) 10s; run: log: (pid 2) 10s\\n'\n"
        "printf 'Interactive shell refcount: 1\\n'\n",
        encoding="utf-8",
    )
    service_script.chmod(0o755)
    monkeypatch.setattr(dashboard, "SERVICES_SCRIPT", service_script)
    with TestClient(dashboard.app) as client:
        resp = client.get("/api/services")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["services"][0]["name"] == "Eidos Scheduler"
        assert payload["services"][0]["running"] is True




def test_services_api_parses_paused_status(monkeypatch, tmp_path: Path) -> None:
    service_script = tmp_path / "eidos_termux_services.sh"
    service_script.write_text(
        "#!/bin/sh\n"
        "printf 'Eidos Scheduler: paused(runit run: /tmp/service: (pid 1) 10s, paused; run: log: (pid 2) 10s)\\n'\n",
        encoding="utf-8",
    )
    service_script.chmod(0o755)
    monkeypatch.setattr(dashboard, "SERVICES_SCRIPT", service_script)
    with TestClient(dashboard.app) as client:
        resp = client.get("/api/services")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["services"][0]["name"] == "Eidos Scheduler"
        assert payload["services"][0]["running"] is False
        assert payload["services"][0]["paused"] is True
def test_services_api_accepts_targeted_restart(monkeypatch) -> None:
    recorded = {}

    async def _fake_service_command(action: str, service: str | None = None):
        recorded["action"] = action
        recorded["service"] = service
        return {"action": action, "service": service or "all", "accepted": True, "queued": True, "ok": True}

    monkeypatch.setattr(dashboard, "_service_command", _fake_service_command)
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/services/restart", params={"service": "atlas"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["queued"] is True
        assert payload["service"] == "atlas"
        assert recorded == {"action": "restart", "service": "atlas"}



def test_services_api_accepts_targeted_pause(monkeypatch) -> None:
    recorded = {}

    async def _fake_service_command(action: str, service: str | None = None):
        recorded["action"] = action
        recorded["service"] = service
        return {"action": action, "service": service or "all", "accepted": True, "queued": True, "ok": True}

    monkeypatch.setattr(dashboard, "_service_command", _fake_service_command)
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/services/pause", params={"service": "scheduler"})
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["queued"] is True
        assert payload["service"] == "scheduler"
        assert recorded == {"action": "pause", "service": "scheduler"}


def test_scheduler_api(monkeypatch, tmp_path: Path) -> None:
    scheduler_script = tmp_path / "eidos_scheduler_control.py"
    scheduler_script.write_text(
        "#!/bin/sh\n"
        'printf \'{"action":"status","state":{"pause_requested":false,"stop_requested":false},"status":{"state":"sleeping","cycle":4,"current_task":"living_pipeline"}}\\n\'\n',
        encoding="utf-8",
    )
    scheduler_script.chmod(0o755)
    monkeypatch.setattr(dashboard, "SCHEDULER_CONTROL_SCRIPT", scheduler_script)
    venv_dir = tmp_path / "eidosian_venv" / "bin"
    venv_dir.mkdir(parents=True, exist_ok=True)
    python_bin = venv_dir / "python"
    python_bin.write_text(
        '#!/bin/sh\nexec "$@"\n',
        encoding="utf-8",
    )
    python_bin.chmod(0o755)
    monkeypatch.setattr(dashboard, "FORGE_ROOT", tmp_path)
    with TestClient(dashboard.app) as client:
        resp = client.get("/api/scheduler")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["payload"]["status"]["state"] == "sleeping"


def test_scheduler_api_invalid_action() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/scheduler/invalid")
        assert resp.status_code == 400


def test_services_api_invalid_target() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/services/restart", params={"service": "bad-target"})
        assert resp.status_code == 400


def test_session_bridge_sync_route(monkeypatch) -> None:
    monkeypatch.setattr(dashboard, "get_session_bridge_status", lambda: {"contract": "eidos.session_bridge.status.v1", "recent_sessions": []})

    def _fake_sync_external_sessions(min_interval_sec: float = 0.0):
        return {"gemini": {"imported": 1}, "codex": {"imported": 2}}

    import eidosian_runtime.session_bridge as bridge_mod

    monkeypatch.setattr(bridge_mod, "sync_external_sessions", _fake_sync_external_sessions)
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/session-bridge/sync")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["sync_result"]["gemini"]["imported"] == 1


def test_system_stats_degrade_partially(monkeypatch) -> None:
    class _Mem:
        percent = 52.0
        used = 4 * 1024**3
        total = 8 * 1024**3

    class _Disk:
        percent = 61.0
        free = 12 * 1024**3

    monkeypatch.setattr(dashboard.psutil, "cpu_percent", lambda interval=None: (_ for _ in ()).throw(PermissionError()))
    monkeypatch.setattr(dashboard.psutil, "virtual_memory", lambda: _Mem())
    monkeypatch.setattr(dashboard.psutil, "disk_usage", lambda path: _Disk())
    monkeypatch.setattr(dashboard.psutil, "boot_time", lambda: 123456)

    payload = dashboard.get_system_stats()
    assert payload["cpu"] is None
    assert payload["ram_percent"] == 52.0
    assert payload["disk_percent"] == 61.0
    assert payload["uptime"] == 123456


def test_consciousness_api_reads_runtime_health(monkeypatch) -> None:
    kernel_module = types.ModuleType("agent_forge.consciousness.kernel")

    class _Kernel:
        @staticmethod
        def read_runtime_health(path):
            return {"beat_count": 7, "path": str(path)}

    kernel_module.ConsciousnessKernel = _Kernel
    monkeypatch.setitem(sys.modules, "agent_forge", types.ModuleType("agent_forge"))
    monkeypatch.setitem(sys.modules, "agent_forge.consciousness", types.ModuleType("agent_forge.consciousness"))
    monkeypatch.setitem(sys.modules, "agent_forge.consciousness.kernel", kernel_module)

    with TestClient(dashboard.app) as client:
        resp = client.get("/api/consciousness")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["beat_count"] == 7


def test_code_forge_provenance_api_defaults(monkeypatch, tmp_path: Path) -> None:
    runtime_dir = tmp_path / "data" / "runtime"
    monkeypatch.setattr(dashboard, "CODE_FORGE_PROVENANCE_AUDIT_STATUS", runtime_dir / "code_forge_provenance_audit_status.json")
    monkeypatch.setattr(dashboard, "CODE_FORGE_PROVENANCE_AUDIT_HISTORY", runtime_dir / "code_forge_provenance_audit_history.jsonl")
    with TestClient(dashboard.app) as client:
        resp = client.get("/api/code-forge/provenance-audit/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "idle"
        hist = client.get("/api/code-forge/provenance-audit/history")
        assert hist.status_code == 200
        assert hist.json()["entries"] == []



def test_services_api_accepts_low_load(monkeypatch) -> None:
    recorded = {}

    async def _fake_service_command(action: str, service: str | None = None):
        recorded["action"] = action
        recorded["service"] = service
        return {"action": action, "service": service or "all", "accepted": True, "queued": True, "ok": True}

    monkeypatch.setattr(dashboard, "_service_command", _fake_service_command)
    with TestClient(dashboard.app) as client:
        resp = client.post("/api/services/low-load")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["queued"] is True
        assert payload["service"] == "all"
        assert recorded == {"action": "low-load", "service": ""}


def test_code_forge_archive_routes(tmp_path: Path, monkeypatch) -> None:
    runtime_dir = tmp_path / "data" / "runtime"
    plan_report_dir = tmp_path / "reports" / "code_forge_archive_plan"
    lifecycle_report_dir = tmp_path / "reports" / "code_forge_archive_lifecycle"
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_PLAN_STATUS", runtime_dir / "code_forge_archive_plan_status.json")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_PLAN_HISTORY", runtime_dir / "code_forge_archive_plan_history.jsonl")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_LIFECYCLE_STATUS", runtime_dir / "code_forge_archive_lifecycle_status.json")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_LIFECYCLE_HISTORY", runtime_dir / "code_forge_archive_lifecycle_history.jsonl")
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_PLAN_REPORT_DIR", plan_report_dir)
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_LIFECYCLE_REPORT_DIR", lifecycle_report_dir)
    monkeypatch.setattr(dashboard, "CODE_FORGE_ARCHIVE_RETIREMENTS_LATEST", tmp_path / "data" / "code_forge" / "archive_ingestion" / "latest" / "retirements" / "latest.json")
    _write_json(runtime_dir / "code_forge_archive_plan_status.json", {"status": "completed", "archive_files_total": 12, "batch_count": 3})
    (runtime_dir / "code_forge_archive_plan_history.jsonl").write_text(json.dumps({"status": "completed"}) + "\n", encoding="utf-8")
    _write_json(runtime_dir / "code_forge_archive_lifecycle_status.json", {"status": "completed", "repo_count": 2})
    (runtime_dir / "code_forge_archive_lifecycle_history.jsonl").write_text(json.dumps({"status": "completed", "phase": "status"}) + "\n", encoding="utf-8")
    _write_json(plan_report_dir / "latest.json", {"archive_files_total": 12, "batch_count": 3})
    _write_json(lifecycle_report_dir / "latest.json", {"repo_count": 2, "summary": {"retirement_ready": 1, "retired": 0}})
    _write_json(tmp_path / "data" / "code_forge" / "archive_ingestion" / "latest" / "retirements" / "latest.json", {"generated_at": "2026-03-21T00:00:00Z", "retirements": []})

    with TestClient(dashboard.app) as client:
        plan = client.get("/api/code-forge/archive-plan")
        assert plan.status_code == 200
        assert plan.json()["status"]["status"] == "completed"
        lifecycle = client.get("/api/code-forge/archive-lifecycle")
        assert lifecycle.status_code == 200
        assert lifecycle.json()["report"]["summary"]["retirement_ready"] == 1


def test_code_forge_archive_wave_route_accepts_retry_failed(monkeypatch) -> None:
    recorded = {}

    def _fake_wave_job(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(dashboard, "_run_code_forge_archive_wave_job", _fake_wave_job)
    monkeypatch.setattr(dashboard.threading.Thread, "start", lambda self: self._target(*self._args, **self._kwargs))

    with TestClient(dashboard.app) as client:
        resp = client.post("/api/code-forge/archive-lifecycle/run-wave?repo_key=eidos_v1_concept&batch_limit=5&retry_failed=true&background=true")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["retry_failed"] is True
        assert recorded["retry_failed"] is True
        assert recorded["repo_keys"] == ["eidos_v1_concept"]


def test_code_forge_archive_preview_prune_and_restore_routes(monkeypatch) -> None:
    recorded = {}

    def _fake_preview_job(**kwargs):
        recorded["preview"] = kwargs

    def _fake_prune_job(**kwargs):
        recorded["prune"] = kwargs

    def _fake_restore_job(**kwargs):
        recorded["restore"] = kwargs

    monkeypatch.setattr(dashboard, "_run_code_forge_archive_preview_job", _fake_preview_job)
    monkeypatch.setattr(dashboard, "_run_code_forge_archive_prune_job", _fake_prune_job)
    monkeypatch.setattr(dashboard, "_run_code_forge_archive_restore_job", _fake_restore_job)
    monkeypatch.setattr(dashboard.threading.Thread, "start", lambda self: self._target(*self._args, **self._kwargs))

    with TestClient(dashboard.app) as client:
        resp = client.post("/api/code-forge/archive-lifecycle/preview-retire?repo_key=eidos_v1_concept&assume_remove_mode=true&background=true")
        assert resp.status_code == 200
        assert resp.json()["phase"] == "preview_retire"
        assert recorded["preview"]["repo_keys"] == ["eidos_v1_concept"]
        assert recorded["preview"]["assume_remove_mode"] is True

        prune = client.post("/api/code-forge/archive-lifecycle/prune-retired?repo_key=eidos_v1_concept&background=true")
        assert prune.status_code == 200
        assert prune.json()["phase"] == "prune_retired"
        assert recorded["prune"]["repo_keys"] == ["eidos_v1_concept"]
        assert recorded["prune"]["dry_run"] is False

        restore = client.post("/api/code-forge/archive-lifecycle/restore?repo_key=eidos_v1_concept&background=true")
        assert restore.status_code == 200
        assert restore.json()["phase"] == "restore"
        assert recorded["restore"]["repo_key"] == "eidos_v1_concept"



def test_build_forge_subprocess_env_includes_code_and_gis_paths(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/custom-path")
    env = dashboard._build_forge_subprocess_env()
    pythonpath = env["PYTHONPATH"].split(":")
    assert str(dashboard.FORGE_ROOT / "code_forge" / "src") in pythonpath
    assert str(dashboard.FORGE_ROOT / "gis_forge" / "src") in pythonpath
    assert env["EIDOS_FORGE_ROOT"] == str(dashboard.FORGE_ROOT)


def test_file_forge_and_shell_apis(monkeypatch, tmp_path: Path) -> None:
    runtime_dir = tmp_path / "data" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    file_forge_db = tmp_path / "data" / "file_forge" / "library.sqlite"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "note.txt").write_text("hello\n", encoding="utf-8")
    FileForge(base_path=tmp_path).index_directory(workspace, db_path=file_forge_db)

    monkeypatch.setattr(dashboard, "FORGE_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "HOME_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(dashboard, "FILE_FORGE_DB", file_forge_db)
    monkeypatch.setattr(dashboard, "FILE_FORGE_INDEX_STATUS", runtime_dir / "file_forge_index_status.json")
    monkeypatch.setattr(dashboard, "FILE_FORGE_INDEX_HISTORY", runtime_dir / "file_forge_index_history.jsonl")

    with TestClient(dashboard.app) as client:
        runtime_resp = client.get("/api/runtime/file-forge")
        assert runtime_resp.status_code == 200
        assert runtime_resp.json()["summary"]["total_files"] == 1

        index_resp = client.post("/api/file-forge/index", params={"background": False, "path": "workspace"})
        assert index_resp.status_code == 200
        assert index_resp.json()["status"] == "completed"

        shell_resp = client.post("/api/shell/start", params={"cwd": ".", "cols": 80, "rows": 24})
        assert shell_resp.status_code == 200
        session_id = shell_resp.json()["session_id"]
        write_resp = client.post(
            "/api/shell/input",
            params={"session_id": session_id, "text": "printf 'hello-shell\n'\nexit\n"},
        )
        assert write_resp.status_code == 200
        time.sleep(0.2)
        read_resp = client.get("/api/shell/read", params={"session_id": session_id, "max_bytes": 8192})
        assert read_resp.status_code == 200
        assert "hello-shell" in read_resp.json().get("output", "")
