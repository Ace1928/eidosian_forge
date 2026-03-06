from knowledge_forge.core.bridge import KnowledgeMemoryBridge


def test_bridge_xref_merges_parallel_links(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    kb_path = tmp_path / "kb.json"

    bridge_a = KnowledgeMemoryBridge(memory_dir=memory_dir, kb_path=kb_path)
    bridge_b = KnowledgeMemoryBridge(memory_dir=memory_dir, kb_path=kb_path)

    assert bridge_a.link_memory_to_knowledge("memory-a", "knowledge-a")
    assert bridge_b.link_memory_to_knowledge("memory-b", "knowledge-b")

    reloaded = KnowledgeMemoryBridge(memory_dir=memory_dir, kb_path=kb_path)
    assert reloaded.memory_to_knowledge == {
        "memory-a": {"knowledge-a"},
        "memory-b": {"knowledge-b"},
    }
    assert reloaded.knowledge_to_memory == {
        "knowledge-a": {"memory-a"},
        "knowledge-b": {"memory-b"},
    }
