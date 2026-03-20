from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

from fastapi.testclient import TestClient
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
            "status": "running",
            "processed": 12,
            "remaining": 4,
            "average_quality_score": 0.88,
            "last_approved": "foo/bar.py",
        },
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
        runtime_dir / "eidos_scheduler_status.json",
        {
            "state": "sleeping",
            "current_task": "living_pipeline",
        },
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
    monkeypatch.setattr(dashboard, "FORGE_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "HOME_ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "RUNTIME_DIR", runtime_dir)
    monkeypatch.setattr(dashboard, "PROOF_REPORT_DIR", tmp_path / "reports" / "proof")
    monkeypatch.setattr(dashboard, "PROOF_BUNDLE_DIR", tmp_path / "reports" / "proof_bundle")
    monkeypatch.setattr(dashboard, "LOCAL_AGENT_STATUS", runtime_dir / "local_mcp_agent" / "status.json")
    monkeypatch.setattr(dashboard, "LOCAL_AGENT_HISTORY", runtime_dir / "local_mcp_agent" / "history.jsonl")
    monkeypatch.setattr(dashboard, "SCHEDULER_STATUS", runtime_dir / "eidos_scheduler_status.json")
    monkeypatch.setattr(dashboard, "COORDINATOR_STATUS", runtime_dir / "forge_coordinator_status.json")
    monkeypatch.setattr(dashboard, "COORDINATOR_HISTORY", runtime_dir / "forge_runtime_trends.json")
    monkeypatch.setattr(dashboard, "DIRECTORY_DOCS_STATUS", runtime_dir / "directory_docs_status.json")
    monkeypatch.setattr(dashboard, "DIRECTORY_DOCS_HISTORY", runtime_dir / "directory_docs_history.json")
    monkeypatch.setattr(dashboard, "SESSION_BRIDGE_CONTEXT", runtime_dir / "session_bridge" / "latest_context.json")
    monkeypatch.setattr(dashboard, "SESSION_BRIDGE_IMPORT_STATUS", runtime_dir / "session_bridge" / "import_status.json")
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
        runtime_resp = client.get("/api/runtime")
        assert runtime_resp.status_code == 200
        runtime_payload = runtime_resp.json()
        assert runtime_payload["local_agent"]["status"] == "success"
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
        local_agent_resp = client.get("/api/runtime/local-agent")
        assert local_agent_resp.status_code == 200
        assert local_agent_resp.json()["status"]["profile"] == "observer"
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
