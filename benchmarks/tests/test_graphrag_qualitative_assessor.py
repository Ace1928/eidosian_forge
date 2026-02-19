from __future__ import annotations

import importlib.util
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
