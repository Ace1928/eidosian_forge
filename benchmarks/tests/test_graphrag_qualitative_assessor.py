from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "graphrag_qualitative_assessor.py"
spec = importlib.util.spec_from_file_location("graphrag_qualitative_assessor", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def test_deterministic_scores_penalize_placeholder_and_empty_query() -> None:
    artifacts = {
        "missing_files": [],
        "stats": {"workflows": {name: {"overall": 0.1} for name in mod.EXPECTED_WORKFLOWS}},
        "expected_entity_hits": ["ALARIC", "SERAPHINA", "KAEL"],
        "entities_count": 14,
        "relationships_count": 91,
        "community_reports_count": 1,
        "community_reports_placeholder": True,
        "query_output": "",
        "index_seconds": 30.0,
        "query_seconds": 1.2,
    }

    scores = mod.deterministic_scores(artifacts)
    assert scores["pipeline_integrity"] == pytest.approx(1.0)
    assert scores["workflow_completeness"] == pytest.approx(1.0)
    assert scores["community_report_quality"] == pytest.approx(0.35)
    assert scores["query_answer_quality"] == pytest.approx(0.1)


def test_aggregate_scores_outputs_rank_and_penalizes_disagreement() -> None:
    deterministic = {
        "pipeline_integrity": 1.0,
        "workflow_completeness": 1.0,
        "entity_coverage": 0.7,
        "relationship_density": 0.8,
        "community_report_quality": 0.9,
        "query_answer_quality": 0.95,
        "runtime_score": 0.7,
    }
    judges = [
        {
            "judge": "a",
            "scores": {
                "factuality": 0.9,
                "grounding": 0.9,
                "coherence": 0.9,
                "usefulness": 0.9,
                "risk_awareness": 0.9,
            },
        },
        {
            "judge": "b",
            "scores": {
                "factuality": 0.3,
                "grounding": 0.3,
                "coherence": 0.3,
                "usefulness": 0.3,
                "risk_awareness": 0.3,
            },
        },
    ]

    agg = mod.aggregate_scores(deterministic, judges)
    assert 0 <= agg["final_score"] <= 1
    assert agg["rank"] in {"A", "B", "C", "D"}
    assert agg["disagreement"] > 0


def test_validate_contract_requires_keys() -> None:
    with pytest.raises(ValueError):
        mod.validate_contract({"contract_version": mod.CONTRACT_VERSION})


def test_aggregate_scores_without_judges_uses_deterministic_baseline() -> None:
    deterministic = {
        "pipeline_integrity": 1.0,
        "workflow_completeness": 1.0,
        "entity_coverage": 1.0,
        "relationship_density": 1.0,
        "community_report_quality": 1.0,
        "query_answer_quality": 1.0,
        "runtime_score": 1.0,
    }
    agg = mod.aggregate_scores(deterministic, [])
    assert agg["judge_score"] == agg["deterministic_objective"]


def test_inconsistent_all_zero_judge_is_rejected_in_high_conf_context() -> None:
    deterministic = {
        "pipeline_integrity": 1.0,
        "workflow_completeness": 1.0,
        "entity_coverage": 0.8,
        "relationship_density": 0.9,
        "community_report_quality": 0.9,
        "query_answer_quality": 0.95,
        "runtime_score": 0.5,
    }
    bad_scores = {
        "factuality": 0.0,
        "grounding": 0.0,
        "coherence": 0.0,
        "usefulness": 0.0,
        "risk_awareness": 0.0,
    }
    assert mod._is_judge_inconsistent(bad_scores, deterministic) is True


def test_aggregate_includes_judge_rejection_counts() -> None:
    deterministic = {
        "pipeline_integrity": 1.0,
        "workflow_completeness": 1.0,
        "entity_coverage": 0.7,
        "relationship_density": 0.8,
        "community_report_quality": 0.9,
        "query_answer_quality": 0.95,
        "runtime_score": 0.7,
    }
    judges = [
        {
            "judge": "a",
            "valid": True,
            "scores": {
                "factuality": 0.9,
                "grounding": 0.9,
                "coherence": 0.9,
                "usefulness": 0.9,
                "risk_awareness": 0.9,
            },
        },
        {
            "judge": "b",
            "valid": False,
            "scores": {
                "factuality": 0.0,
                "grounding": 0.0,
                "coherence": 0.0,
                "usefulness": 0.0,
                "risk_awareness": 0.0,
            },
        },
    ]

    agg = mod.aggregate_scores(deterministic, judges)
    assert agg["judge_total"] == 2
    assert agg["judge_valid"] == 1
    assert agg["judge_rejected"] == 1


def test_normalize_judge_payload_requires_verdict_and_risk_array() -> None:
    payload = mod._normalize_judge_payload(
        {
            "scores": {
                "factuality": 1.2,
                "grounding": 0.8,
                "coherence": 0.7,
                "usefulness": 0.9,
                "risk_awareness": -1,
            },
            "verdict": "Grounded and useful.",
            "risks": ["latency drift", ""],
        }
    )
    assert payload is not None
    assert payload["scores"]["factuality"] == 1.0
    assert payload["scores"]["risk_awareness"] == 0.0
    assert payload["risks"] == ["latency drift"]

    assert mod._normalize_judge_payload({"scores": {}, "verdict": "", "risks": []}) is None
    assert mod._normalize_judge_payload({"scores": {}, "verdict": "ok", "risks": "bad"}) is None


def test_judge_prompt_embeds_schema() -> None:
    prompt = mod._judge_prompt(
        {"entities_count": 1, "relationships_count": 2, "community_reports_placeholder": False, "expected_entity_hits": [], "query_output": "ok"},
        {"pipeline_integrity": 1.0},
        [],
    )
    assert '"required": ["scores", "verdict", "risks"]' in prompt
    assert "risk_awareness" in prompt


def test_load_artifacts_uses_native_reports_without_pandas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_dir = tmp_path / "workspace" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stats.json").write_text(
        json.dumps({"workflows": {name: {"overall": 0.2} for name in mod.EXPECTED_WORKFLOWS}}),
        encoding="utf-8",
    )
    (output_dir / "native_community_reports.json").write_text(
        json.dumps(
            {
                "aggregate": {"average_quality_score": 0.82},
                "reports": [
                    {
                        "community": "documents",
                        "title": "Alaric and Seraphina in Eidos",
                        "summary": "Alaric and Seraphina coordinate defense of the Crystal in Eidos.",
                        "findings": ["Kael protects the Crystal from Malakar near the Whispering Caves."],
                        "node_ids": ["n1", "n2", "n3"],
                        "metrics": {"link_density": 2.5, "quality_score": 0.82},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(mod, "_require_pandas", lambda: (_ for _ in ()).throw(AssertionError("pandas should not be required")))
    artifacts = mod.load_artifacts(tmp_path / "workspace", None)

    assert artifacts["artifact_mode"] == "native_json"
    assert artifacts["community_reports_count"] == 1
    assert artifacts["community_reports_placeholder"] is False
    assert artifacts["native_average_quality_score"] == pytest.approx(0.82)
    assert "ALARIC" in artifacts["expected_entity_hits"]
    assert "SERAPHINA" in artifacts["expected_entity_hits"]


def test_deterministic_scores_use_native_quality_and_link_density() -> None:
    artifacts = {
        "artifact_mode": "native_json",
        "missing_files": [],
        "stats": {"workflows": {name: {"overall": 0.1} for name in mod.EXPECTED_WORKFLOWS}},
        "expected_entity_hits": mod.EXPECTED_ENTITIES,
        "entities_count": 8,
        "relationships_count": 0,
        "community_reports_count": 2,
        "community_reports_placeholder": False,
        "native_average_quality_score": 0.81,
        "native_average_link_density": 2.4,
        "query_output": "Kael and Seraphina protect the Crystal of Eternity in Eidos.",
        "index_seconds": 12.0,
        "query_seconds": 1.0,
    }

    scores = mod.deterministic_scores(artifacts)
    assert scores["community_report_quality"] == pytest.approx(0.81)
    assert scores["relationship_density"] == pytest.approx(0.6)
