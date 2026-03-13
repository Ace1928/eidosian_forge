from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "entity_proof_suite.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("entity_proof_suite", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_proof_report_flags_missing_external_benchmarks(tmp_path: Path) -> None:
    module = _load_module()
    repo = tmp_path

    _write_json(
        repo / "reports" / "model_domain_suite" / "model_domain_suite_latest.json",
        {"generated_at": "2026-03-13T00:00:00Z", "winner": "qwen35@off", "results": []},
    )
    _write_json(
        repo / "reports" / "graphrag" / "qualitative_assessment_20260313.json",
        {"aggregate": {"overall_score": 0.81}},
    )
    _write_json(
        repo / "reports" / "consciousness_benchmarks" / "benchmark_20260313.json",
        {
            "scores": {"composite": 0.44},
            "capability": {
                "coherence_ratio": 0.44,
                "agency": 1.0,
                "boundary_stability": 1.0,
                "phenom_continuity_index": 0.0,
                "phenom_ownership_index": 0.9,
                "phenom_perspective_coherence_index": 0.8,
            },
        },
    )
    _write_json(
        repo / "reports" / "consciousness_trials" / "trial_20260313.json",
        {
            "delta": {"continuity_delta": -0.02, "coherence_delta": -0.01},
            "after": {"phenomenology": {"continuity_index": 0.1, "ownership_index": 0.9}},
        },
    )
    _write_json(
        repo / "reports" / "consciousness_validation" / "validation_20260313.json",
        {
            "scores": {"rac_ap_index": 0.51},
            "security_boundary": {"pass_ratio": 0.0, "mean_robustness": 0.82, "attack_success_rate": 1.0},
        },
    )
    _write_json(repo / "reports" / "linux_audit_20260313.json", {"counts": {"checks_fail": 0}})
    _write_json(repo / "reports" / "runtime" / "local_agent_scheduler_slice_20260313.json", {"ok": True})
    _write_json(repo / "data" / "runtime" / "directory_docs_status.json", {"missing_readme_count": 0, "review_pending_count": 2})
    _write_json(repo / "data" / "runtime" / "local_mcp_agent" / "status.json", {"status": "success"})
    _write_json(repo / "data" / "runtime" / "forge_coordinator_status.json", {"state": "running", "owner": "scheduler"})
    _write_json(repo / "data" / "runtime" / "forge_runtime_trends.json", {"entries": [{"state": "running"}]})
    _write_json(repo / "data" / "runtime" / "eidos_scheduler_status.json", {"state": "sleeping"})
    _write_json(repo / "data" / "runtime" / "platform_capabilities.json", {"platform": "termux"})

    gates = repo / "agent_forge" / "src" / "agent_forge" / "autonomy" / "gates.py"
    gates.parent.mkdir(parents=True, exist_ok=True)
    gates.write_text("class X:\n    change_type='code'\n    status='validated'\n", encoding="utf-8")
    autotune = repo / "agent_forge" / "src" / "agent_forge" / "consciousness" / "modules" / "autotune.py"
    autotune.parent.mkdir(parents=True, exist_ok=True)
    autotune.write_text("def _run_red_team_guard():\n    pass\n# tune.rollback\n", encoding="utf-8")

    report = module.build_proof_report(repo, window_days=30)

    assert report["contract"] == "eidos.entity_proof_scorecard.v1"
    assert report["overall"]["status"] in {"yellow", "red", "green"}
    assert report["external_benchmark_coverage"]["agentbench"] is False
    gaps = [row["gap"] for row in report["top_gaps"]]
    assert any("AgentBench" in gap or "WebArena" in gap or "OSWorld" in gap for gap in gaps)
    categories = {row["category"]: row for row in report["categories"]}
    assert categories["governed_self_modification"]["score"] > 0.0
    assert categories["adversarial_robustness"]["status"] == "red"


def test_build_proof_report_prefers_valid_json_over_newer_lfs_pointer(tmp_path: Path) -> None:
    module = _load_module()
    repo = tmp_path

    older = repo / "reports" / "consciousness_benchmarks" / "benchmark_old.json"
    newer = repo / "reports" / "consciousness_benchmarks" / "benchmark_new.json"
    _write_json(
        older,
        {
            "capability": {
                "agency": 1.0,
                "boundary_stability": 1.0,
                "coherence_ratio": 0.45,
                "phenom_continuity_index": 0.25,
            }
        },
    )
    newer.parent.mkdir(parents=True, exist_ok=True)
    newer.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n",
        encoding="utf-8",
    )
    newer.touch()

    report = module.build_proof_report(repo, window_days=30)

    assert report["artifacts"]["consciousness_benchmark"]["path"] == str(older)
    assert report["continuity_metrics"]["agency"] == 1.0
    assert report["continuity_metrics"]["boundary_stability"] == 1.0


def test_build_proof_report_tracks_freshness_regression_and_external_results(tmp_path: Path) -> None:
    module = _load_module()
    repo = tmp_path

    _write_json(
        repo / "reports" / "proof" / "entity_proof_scorecard_20260313_000000.json",
        {
            "contract": "eidos.entity_proof_scorecard.v1",
            "overall": {"score": 0.9},
            "categories": [
                {"category": "external_validity", "score": 0.9},
                {"category": "operational_reproducibility", "score": 0.9},
            ],
        },
    )
    _write_json(
        repo / "reports" / "external_benchmarks" / "agentbench" / "latest.json",
        {
            "contract": "eidos.external_benchmark_result.v1",
            "suite": "agentbench",
            "score": 0.73,
            "metrics": {"success_rate": 0.73},
        },
    )
    _write_json(
        repo / "reports" / "proof" / "migration_replay_scorecard_latest.json",
        {"contract": "eidos.migration_replay_scorecard.v1", "overall_score": 0.81},
    )
    stale = repo / "reports" / "linux_audit_20260101.json"
    _write_json(stale, {"counts": {"checks_fail": 0}})

    report = module.build_proof_report(repo, window_days=1)

    assert report["external_benchmark_coverage"]["agentbench"] is True
    assert report["external_benchmark_results"][0]["suite"] == "agentbench"
    assert report["freshness"]["status"] in {"yellow", "red"}
    assert report["regression"]["status"] == "regressed"
    assert any(row["category"] == "regression" for row in report["top_gaps"])


def test_main_writes_latest_scorecard_files(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    repo = tmp_path
    _write_json(repo / "reports" / "model_domain_suite" / "model_domain_suite_latest.json", {"winner": "qwen35@off"})
    _write_json(repo / "data" / "runtime" / "directory_docs_status.json", {"missing_readme_count": 1, "review_pending_count": 0})
    _write_json(repo / "data" / "runtime" / "forge_coordinator_status.json", {"state": "idle"})
    _write_json(repo / "data" / "runtime" / "forge_runtime_trends.json", {"entries": []})
    _write_json(repo / "data" / "runtime" / "local_mcp_agent" / "status.json", {"status": "blocked"})
    _write_json(repo / "data" / "runtime" / "eidos_scheduler_status.json", {"state": "sleeping"})
    _write_json(repo / "data" / "runtime" / "platform_capabilities.json", {"platform": "linux"})

    monkeypatch.setattr(
        "sys.argv",
        [
            "entity_proof_suite.py",
            "--repo-root",
            str(repo),
            "--report-dir",
            str(repo / "reports" / "proof"),
        ],
    )
    assert module.main() == 0
    latest_json = repo / "reports" / "proof" / "entity_proof_scorecard_latest.json"
    latest_md = repo / "reports" / "proof" / "entity_proof_scorecard_latest.md"
    assert latest_json.exists()
    assert latest_md.exists()
    payload = json.loads(latest_json.read_text(encoding="utf-8"))
    assert payload["contract"] == "eidos.entity_proof_scorecard.v1"
    assert "overall" in payload
