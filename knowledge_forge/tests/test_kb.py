import pytest
from pathlib import Path
from knowledge_forge import KnowledgeForge, GraphRAGIntegration

def test_kb_graph(tmp_path):
    kf = KnowledgeForge(persistence_path=tmp_path / "kb.json")
    n1 = kf.add_knowledge("Entity A", concepts=["concept1"])
    n2 = kf.add_knowledge("Entity B", concepts=["concept1"])
    
    kf.link_nodes(n1.id, n2.id)
    
    assert len(kf.get_by_concept("concept1")) == 2
    assert n2.id in kf.get_related_nodes(n1.id)[0].id

def test_graphrag_integration(tmp_path):
    grag = GraphRAGIntegration(graphrag_root=tmp_path)
    res = grag.global_query("test")
    assert res["success"]
    assert "graphrag.query" in res["command"][2]
