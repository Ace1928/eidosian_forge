from __future__ import annotations

import pytest

from knowledge_forge import KnowledgeForge

pytest.importorskip("rdflib", reason="rdflib required for RDF tests")

def test_rdf_export_import_roundtrip(tmp_path):
    source_path = tmp_path / "kb_source.json"
    rdf_path = tmp_path / "kb.ttl"
    target_path = tmp_path / "kb_target.json"

    kb = KnowledgeForge(persistence_path=source_path)
    node_a = kb.add_knowledge("Agent forge handles workspace competition", concepts=["agent_forge"], tags=["agent"])
    node_b = kb.add_knowledge("Memory forge now compresses old memories", concepts=["memory_forge"], tags=["memory"])
    kb.link_nodes(node_a.id, node_b.id)

    export_report = kb.export_rdf(rdf_path, format="turtle")
    assert export_report["node_count"] == 2
    assert export_report["triple_count"] > 0
    assert rdf_path.exists()

    imported = KnowledgeForge(persistence_path=target_path)
    import_report = imported.import_rdf(rdf_path, format="turtle")
    assert import_report["imported_nodes"] == 2
    assert imported.stats()["node_count"] == 2
    assert imported.get_by_concept("agent_forge")
    related = imported.get_related_nodes(node_a.id)
    assert related
    assert related[0].id == node_b.id
