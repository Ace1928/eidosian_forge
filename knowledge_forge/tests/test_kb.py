import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from knowledge_forge import GraphRAGIntegration, KnowledgeForge


def test_kb_graph(tmp_path):
    kf = KnowledgeForge(persistence_path=tmp_path / "kb.json")
    n1 = kf.add_knowledge("Entity A", concepts=["concept1"])
    n2 = kf.add_knowledge("Entity B", concepts=["concept1"])

    kf.link_nodes(n1.id, n2.id)

    assert len(kf.get_by_concept("concept1")) == 2
    assert n2.id in kf.get_related_nodes(n1.id)[0].id


@pytest.mark.skip(reason="Integration test requiring graphrag CLI setup")
def test_graphrag_integration(tmp_path):
    grag = GraphRAGIntegration(graphrag_root=tmp_path)
    res = grag.global_query("test")
    assert res["success"]
    command_text = " ".join(str(part) for part in res["command"])
    assert "graphrag" in command_text
    assert "query" in command_text


def test_graphrag_integration_command(tmp_path):
    """Test that GraphRAGIntegration builds correct commands."""

    grag = GraphRAGIntegration(graphrag_root=tmp_path / "grag")
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "ok"
        run_mock.return_value.stderr = ""
        res = grag.global_query("test query")
    # Verify command structure (don't execute)
    command_text = " ".join(str(part) for part in res["command"])
    assert "graphrag" in command_text
    assert "query" in command_text
    assert "--method" in res["command"]
    assert "global" in res["command"]


def test_graphrag_legacy_fallback_command(tmp_path):
    """GraphRAGIntegration falls back to legacy module style when needed."""

    grag = GraphRAGIntegration(graphrag_root=tmp_path / "grag")
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        first = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag", "query"],
            returncode=1,
            stdout="",
            stderr="No module named graphrag.__main__",
        )
        second = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag.query"],
            returncode=0,
            stdout="ok",
            stderr="",
        )
        run_mock.side_effect = [first, second]
        res = grag.global_query("test query")

    command_text = " ".join(str(part) for part in res["command"])
    assert "graphrag.query" in command_text
    assert res["fallback_used"] is True


def test_graphrag_legacy_fallback_does_not_override_failed_primary(tmp_path):
    """When fallback also fails, keep primary command result for clearer diagnostics."""

    grag = GraphRAGIntegration(graphrag_root=tmp_path / "grag")
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        first = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag", "query"],
            returncode=1,
            stdout="",
            stderr="No module named graphrag.__main__",
        )
        second = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag.query"],
            returncode=2,
            stdout="",
            stderr="other error",
        )
        run_mock.side_effect = [first, second]
        res = grag.global_query("test query")

    command_text = " ".join(str(part) for part in res["command"])
    assert " -m graphrag " in f" {command_text} "
    assert res["fallback_used"] is True
    assert res["success"] is False


def test_graphrag_timeout_returns_structured_error(tmp_path):
    """Timeouts should return deterministic diagnostics instead of hanging callers."""

    grag = GraphRAGIntegration(graphrag_root=tmp_path / "grag")
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        run_mock.side_effect = subprocess.TimeoutExpired(
            cmd=["python", "-m", "graphrag", "query"],
            timeout=grag.timeout_seconds,
            output="",
            stderr="timeout",
        )
        res = grag.global_query("test query")

    assert res["success"] is False
    assert res["returncode"] == 124
    assert "timed out" in res["stderr"].lower()


class _FakeNode:
    def __init__(self, node_id: str, content: str, tags: list[str] | None = None) -> None:
        self.id = node_id
        self.content = content
        self.tags = set(tags or [])


class _FakeKnowledge:
    def get_related_nodes(self, node_id: str):
        if node_id == "kb-1":
            return [_FakeNode("kb-2", "Related graph evidence", ["graph"])]
        return []

    def find_path(self, start_id: str, end_id: str):
        if {start_id, end_id} == {"kb-1", "kb-3"}:
            return ["kb-1", "kb-2", "kb-3"]
        return []


class _FakeBridge:
    def __init__(self) -> None:
        self.knowledge = _FakeKnowledge()

    def get_memory_knowledge_context(self, query: str, max_results: int = 8):
        _ = query
        return {
            "query": query,
            "total_results": 3,
            "memory_context": [{"id": "mem-1", "content": "Relevant memory", "score": 0.8}],
            "knowledge_context": [
                {"id": "kb-1", "content": "Primary knowledge node", "score": 0.9, "tags": ["root"]},
                {"id": "kb-3", "content": "Secondary knowledge node", "score": 0.85, "tags": ["leaf"]},
            ][:max_results],
        }


def test_graphrag_query_falls_back_to_local_vector_graph_context(tmp_path):
    grag = GraphRAGIntegration(graphrag_root=tmp_path / "grag", bridge=_FakeBridge())
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag", "query"],
            returncode=1,
            stdout="",
            stderr="No module named graphrag.__main__",
        )
        res = grag.local_query("vector graph query")

    assert res["success"] is True
    assert res["fallback_used"] is True
    assert res["local_fallback"] is True
    assert res["mode"] == "local_vector_graph"
    assert res["knowledge_context"][0]["id"] == "kb-1"
    assert res["graph_neighbors"][0]["id"] == "kb-2"
    assert res["graph_paths"][0] == ["kb-1", "kb-2", "kb-3"]


def test_graphrag_incremental_index_native_fallback_ingests_docs_and_word_graph(tmp_path):
    workspace = tmp_path / "workspace"
    docs = workspace / "input"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "guide.md").write_text("Eidos uses a unified vector graph index.", encoding="utf-8")
    artifact_dir = tmp_path / "data" / "code_forge" / "cycle" / "run_001"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "provenance_registry.json").write_text(
        """
        {
          "schema_version": "code_forge_provenance_registry_v1",
          "generated_at": "2026-03-06T00:00:00+00:00",
          "registry_id": "reg_1",
          "stage": "archive_digester",
          "root_path": "/tmp/repo",
          "provenance_id": "prov_1",
          "integration_policy": "effective_run",
          "integration_run_id": "run_1",
          "artifacts": [{"artifact_kind": "triage", "path": "/tmp/out/triage.json"}],
          "links": {
            "knowledge_count": 1,
            "memory_count": 0,
            "graphrag_count": 1,
            "unit_links": [
              {"unit_id": "u1", "qualified_name": "pkg.mod.fn", "knowledge_node_id": "", "memory_id": ""}
            ]
          },
          "benchmark": {"gate_pass": true, "search_p95_ms": 42.0},
          "drift": {"warning_count": 1, "max_abs_delta": 3.0}
        }
        """.strip(),
        encoding="utf-8",
    )
    word_graph = tmp_path / "eidos_semantic_graph.json"
    word_graph.write_text(
        """
        {
          "nodes": [
            {"id": "autonomy", "definition": "self-directed action"},
            {"id": "agency", "definition": "capacity to act"}
          ],
          "edges": [
            {"source": "autonomy", "target": "agency", "type": "related", "weight": 0.9}
          ]
        }
        """.strip(),
        encoding="utf-8",
    )

    grag = GraphRAGIntegration(
        graphrag_root=workspace,
        kb_path=tmp_path / "kb.json",
        memory_dir=tmp_path / "memory",
        word_graph_path=word_graph,
    )
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag", "index"],
            returncode=2,
            stdout="",
            stderr="graphrag not installed",
        )
        result = grag.run_incremental_index([docs, artifact_dir])

    assert result["success"] is True
    assert result["mode"] == "native_vector_graph"
    assert result["external_success"] is False
    assert result["files_indexed"] >= 1
    assert result["word_forge"]["term_nodes"] == 2
    assert result["word_forge"]["relationships"] == 1
    assert result["code_forge"]["artifacts_indexed"] >= 1
    assert result["community_reports"]["count"] >= 3

    knowledge = KnowledgeForge(persistence_path=tmp_path / "kb.json")
    assert knowledge.search("Eidos uses a unified vector graph index.")
    word_hits = knowledge.search("Word Forge term: autonomy")
    assert word_hits
    related = knowledge.get_related_nodes(word_hits[0].id)
    assert related
    artifact_hits = knowledge.search("Code Forge artifact:")
    assert artifact_hits


def test_graphrag_incremental_index_removes_stale_native_documents(tmp_path):
    workspace = tmp_path / "workspace"
    docs = workspace / "input"
    docs.mkdir(parents=True, exist_ok=True)
    doc_path = docs / "note.txt"
    doc_path.write_text("persistent graph memory", encoding="utf-8")

    grag = GraphRAGIntegration(
        graphrag_root=workspace,
        kb_path=tmp_path / "kb.json",
        memory_dir=tmp_path / "memory",
        word_graph_path=tmp_path / "missing_word_graph.json",
    )
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag", "index"],
            returncode=2,
            stdout="",
            stderr="graphrag not installed",
        )
        first = grag.run_incremental_index([docs])
        doc_path.unlink()
        second = grag.run_incremental_index([docs])

    assert first["files_indexed"] == 1
    assert second["files_removed"] == 1

    knowledge = KnowledgeForge(persistence_path=tmp_path / "kb.json")
    assert knowledge.search("persistent graph memory") == []


def test_graphrag_native_reports_written_after_index(tmp_path):
    workspace = tmp_path / "workspace"
    docs = workspace / "input"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "architecture.md").write_text(
        "Vector graph architecture links memory, code, and knowledge.", encoding="utf-8"
    )

    grag = GraphRAGIntegration(
        graphrag_root=workspace,
        kb_path=tmp_path / "kb.json",
        memory_dir=tmp_path / "memory",
        word_graph_path=tmp_path / "missing_word_graph.json",
    )
    with patch("knowledge_forge.integrations.graphrag.subprocess.run") as run_mock:
        run_mock.return_value = subprocess.CompletedProcess(
            args=["python", "-m", "graphrag", "index"],
            returncode=2,
            stdout="",
            stderr="graphrag not installed",
        )
        result = grag.run_incremental_index([docs])

    reports = result["community_reports"]
    assert reports["count"] >= 1
    assert reports["average_quality_score"] > 0
    assert Path(reports["json_path"]).exists()
    assert Path(reports["markdown_path"]).exists()
    payload = json.loads(Path(reports["json_path"]).read_text(encoding="utf-8"))
    assert payload["aggregate"]["average_quality_score"] > 0
    assert payload["reports"]
    summary = grag.native_report_summary(limit=2)
    assert summary["count"] >= 1
    assert summary["average_quality_score"] > 0
    trends = grag.native_trend_summary(limit=5)
    assert trends["count"] >= 1
    assert trends["latest"]["average_quality_score"] > 0
    artifact_summary = grag.native_artifact_summary(limit=2)
    assert artifact_summary["count"] >= 0
    assert "artifacts" in artifact_summary
