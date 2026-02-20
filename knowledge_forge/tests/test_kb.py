from pathlib import Path

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
    assert "graphrag.query" in res["command"][2]


def test_graphrag_integration_command():
    """Test that GraphRAGIntegration builds correct commands."""

    grag = GraphRAGIntegration(graphrag_root=Path("/tmp/test"))
    res = grag.global_query("test query")
    # Verify command structure (don't execute)
    assert "graphrag.query" in res["command"][2]
    assert "--method" in res["command"]
    assert "global" in res["command"]
