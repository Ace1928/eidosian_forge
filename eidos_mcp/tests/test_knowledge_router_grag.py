from __future__ import annotations

import json

from eidos_mcp.routers import knowledge


class _FakeGraphRAG:
    def native_report_summary(self, limit: int = 5):
        return {
            "count": 2,
            "reports": [
                {
                    "community": "documents",
                    "title": "Documents Community",
                    "summary": "linked docs and memory context",
                    "rating": 4,
                    "finding_count": 2,
                    "node_count": 3,
                }
            ][:limit],
        }

    def native_artifact_summary(self, limit: int = 10):
        return {
            "count": 1,
            "kinds": {"code_forge_provenance_registry": 1},
            "items": [
                {
                    "kind": "code_forge_provenance_registry",
                    "artifact_path": "data/code_forge/cycle/run_001/provenance_registry.json",
                    "detail_node_count": 2,
                }
            ][:limit],
        }


def test_grag_reports_tool_returns_json(monkeypatch) -> None:
    monkeypatch.setattr(knowledge, "grag", _FakeGraphRAG())
    payload = json.loads(knowledge.grag_reports(limit=1))
    assert payload["count"] == 2
    assert payload["reports"][0]["community"] == "documents"


def test_grag_artifacts_tool_returns_json(monkeypatch) -> None:
    monkeypatch.setattr(knowledge, "grag", _FakeGraphRAG())
    payload = json.loads(knowledge.grag_artifacts(limit=1))
    assert payload["count"] == 1
    assert payload["items"][0]["kind"] == "code_forge_provenance_registry"
