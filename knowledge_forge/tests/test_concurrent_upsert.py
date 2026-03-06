from knowledge_forge import KnowledgeForge


def test_add_knowledge_upserts_same_content_across_instances(tmp_path) -> None:
    kb_path = tmp_path / "kb.json"
    kb_a = KnowledgeForge(persistence_path=kb_path)
    kb_b = KnowledgeForge(persistence_path=kb_path)

    first = kb_a.add_knowledge(
        "Tiered memory writes should be atomic and duplicate-resistant.",
        concepts=["memory"],
        tags=["atomic"],
        metadata={"source": "a", "nested": {"safe": True}},
    )
    second = kb_b.add_knowledge(
        "Tiered memory writes should be atomic and duplicate-resistant.",
        concepts=["reliability"],
        tags=["dedup"],
        metadata={"source": "b", "nested": {"merge": True}},
    )

    assert second.id == first.id

    reloaded = KnowledgeForge(persistence_path=kb_path)
    assert len(reloaded.nodes) == 1
    node = next(iter(reloaded.nodes.values()))
    assert node.tags == {"atomic", "dedup"}
    assert set(reloaded.concept_map) == {"memory", "reliability"}
    assert node.metadata["nested"] == {"safe": True, "merge": True}

