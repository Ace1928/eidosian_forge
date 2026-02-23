from pathlib import Path

import pytest

from knowledge_forge import KnowledgeForge

rdflib = pytest.importorskip("rdflib", reason="rdflib required for reasoning tests")
pytest.importorskip("owlrl", reason="owlrl required for reasoning tests")



def test_reason_rdf_graph_rdfs_infers_superclass_type() -> None:
    graph = rdflib.Graph()
    ex = rdflib.Namespace("urn:test:")

    graph.add((ex.Child, rdflib.RDFS.subClassOf, ex.Parent))
    graph.add((ex.instance, rdflib.RDF.type, ex.Child))

    report = KnowledgeForge.reason_rdf_graph(graph, profile="rdfs")

    assert report["profile"] == "rdfs"
    assert report["triple_count_after"] >= report["triple_count_before"]
    assert (ex.instance, rdflib.RDF.type, ex.Parent) in graph



def test_reason_owl_can_export_and_apply(tmp_path: Path) -> None:
    kb = KnowledgeForge(persistence_path=tmp_path / "kb.json")
    kb.add_knowledge("Agent has memory", concepts=["agent"], tags=["memory"])

    out = tmp_path / "reasoned.ttl"
    report = kb.reason_owl(profile="owlrl", apply=True, output_path=out, output_format="turtle")

    assert report["applied"] is True
    assert report["triple_count_after"] >= report["triple_count_before"]
    assert report["imported_nodes"] >= 1
    assert out.exists()
