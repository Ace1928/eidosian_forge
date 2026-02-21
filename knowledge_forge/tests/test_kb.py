import subprocess
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
