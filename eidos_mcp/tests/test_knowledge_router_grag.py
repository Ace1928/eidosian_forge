from __future__ import annotations

import json
from pathlib import Path

from eidos_mcp.routers import knowledge


def test_knowledge_router_grag_report_tools(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    output_dir = workspace / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "native_community_reports.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-06T03:05:00+00:00",
                "aggregate": {
                    "average_quality_score": 0.61,
                    "average_rating": 3.5,
                    "weak_communities": 1,
                    "top_community": "documents",
                },
                "reports": [
                    {
                        "community": "documents",
                        "title": "Documents Community",
                        "summary": "4 nodes with linked context",
                        "rating": 4,
                        "metrics": {
                            "quality_score": 0.61,
                            "quality_band": "moderate",
                            "coverage_ratio": 0.5,
                            "link_density": 1.0,
                        },
                        "findings": ["Vector graph architecture"],
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "native_graphrag_state.json").write_text(
        json.dumps(
            {
                "files": {},
                "word_graph": {},
                "code_forge_artifacts": {
                    "artifact_1": {
                        "artifact_path": "reports/code_forge/triage.json",
                        "kind": "triage",
                        "benchmark_gate_pass": True,
                        "drift_warning_count": 0,
                        "updated_at": "2026-03-06T03:00:00+00:00",
                    }
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(knowledge, "grag", knowledge.GraphRAGIntegration(graphrag_root=Path(workspace)))

    report_payload = json.loads(knowledge.grag_reports(limit=2))
    assert report_payload["count"] == 1
    assert report_payload["average_quality_score"] == 0.61
    assert report_payload["reports"][0]["community"] == "documents"

    quality_payload = json.loads(knowledge.grag_report_quality(limit=2))
    assert quality_payload["top_community"] == "documents"
    assert quality_payload["weak_communities"] == 1

    artifact_payload = json.loads(knowledge.grag_artifacts(limit=2))
    assert artifact_payload["count"] == 1
    assert artifact_payload["artifacts"][0]["kind"] == "triage"
