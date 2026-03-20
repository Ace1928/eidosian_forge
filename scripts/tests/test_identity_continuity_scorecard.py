from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "identity_continuity_scorecard.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("identity_continuity_scorecard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_identity_scorecard(tmp_path: Path) -> None:
    module = _load_module()
    repo = tmp_path
    _write_json(
        repo / "reports" / "consciousness_benchmarks" / "benchmark_20260320.json",
        {
            "capability": {
                "agency": 1.0,
                "boundary_stability": 1.0,
                "phenom_continuity_index": 0.3,
                "phenom_ownership_index": 0.8,
                "phenom_perspective_coherence_index": 0.7,
            }
        },
    )
    _write_json(
        repo / "reports" / "consciousness_trials" / "trial_20260320.json",
        {
            "delta": {"continuity_delta": -0.02, "coherence_delta": -0.01},
            "after": {"phenomenology": {"continuity_index": 0.4, "ownership_index": 0.85}},
        },
    )
    _write_json(repo / "reports" / "proof" / "entity_proof_scorecard_latest.json", {"overall": {"score": 0.7}})
    _write_json(
        repo / "data" / "runtime" / "session_bridge" / "latest_context.json",
        {"recent_sessions": [{"session_id": "codex:x"}, {"session_id": "gemini:y"}]},
    )
    _write_json(
        repo / "data" / "runtime" / "session_bridge" / "import_status.json",
        {"gemini": {"imported_ids": ["g1"]}, "codex": {"threads": {"t1": 1, "t2": 2}}},
    )
    _write_json(repo / "data" / "tiered_memory" / "working.json", [{"id": "m1"}, {"id": "m2"}])
    (repo / "docs" / "THEORY_OF_OPERATION.md").parent.mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "THEORY_OF_OPERATION.md").write_text("# theory\n", encoding="utf-8")
    (repo / "EIDOS_IDENTITY_MANIFESTO.md").write_text("# manifesto\n", encoding="utf-8")
    ledger = repo / "data" / "runtime" / "test_autonomy" / "ledger" / "continuity_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text('{"event":"tick"}\n', encoding="utf-8")
    _write_json(
        repo / "reports" / "proof" / "identity_continuity_scorecard_20260319_000000.json",
        {"contract": "eidos.identity_continuity_scorecard.v1", "generated_at": "2026-03-19T00:00:00Z", "overall_score": 0.7, "status": "yellow"},
    )

    payload = module.build_scorecard(repo)
    assert payload["contract"] == "eidos.identity_continuity_scorecard.v1"
    assert payload["overall_score"] > 0.5
    assert payload["history"]["trend"] == "improved"
    assert payload["history"]["sample_count"] == 2
    assert payload["session_bridge"]["imported_codex_threads"] == 2
    assert payload["memory"]["working_records"] == 2
    assert payload["identity_sources"]["theory_of_operation"] is True


def test_main_writes_latest_files(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    repo = tmp_path
    (repo / "docs").mkdir(parents=True, exist_ok=True)
    (repo / "docs" / "THEORY_OF_OPERATION.md").write_text("# theory\n", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "identity_continuity_scorecard.py",
            "--repo-root",
            str(repo),
            "--report-dir",
            str(repo / "reports" / "proof"),
        ],
    )
    assert module.main() == 0
    assert (repo / "reports" / "proof" / "identity_continuity_scorecard_latest.json").exists()
    assert (repo / "reports" / "proof" / "identity_continuity_scorecard_latest.md").exists()
