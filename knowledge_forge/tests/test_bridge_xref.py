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


class _FakeEmbedder:
    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        if "latency" in lowered or "benchmark" in lowered:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]


def test_bridge_unified_search_uses_semantic_backends(tmp_path) -> None:
    memory_dir = tmp_path / "memory"
    kb_path = tmp_path / "kb.json"
    bridge = KnowledgeMemoryBridge(memory_dir=memory_dir, kb_path=kb_path, embedder=_FakeEmbedder())

    memory_id = bridge.memory.remember("Latency benchmark memory")
    node = bridge.knowledge.add_knowledge("Latency benchmark reference", tags=["benchmark"])

    results = bridge.unified_search("latency benchmark", limit=4)

    assert any(result.source == "memory" and result.id == memory_id for result in results)
    assert any(result.source == "knowledge" and result.id == node.id for result in results)
