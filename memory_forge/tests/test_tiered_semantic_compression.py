from __future__ import annotations

from datetime import datetime, timedelta

from memory_forge.core.interfaces import MemoryType
from memory_forge.core.tiered_memory import MemoryNamespace, MemoryTier, TieredMemorySystem


def _age_item(memory: TieredMemorySystem, memory_id: str, days: int) -> None:
    item = memory.tiers[MemoryTier.LONG_TERM][memory_id]
    ts = datetime.now() - timedelta(days=days)
    item.created_at = ts
    item.last_accessed = ts


def test_semantic_compress_old_memories_creates_summary_and_marks_sources(tmp_path) -> None:
    memory = TieredMemorySystem(persistence_dir=tmp_path)

    source_ids = [
        memory.remember(
            "Agent forge benchmark run improved ignition trace stability and reduced latency.",
            tier=MemoryTier.LONG_TERM,
            namespace=MemoryNamespace.KNOWLEDGE,
            memory_type=MemoryType.EPISODIC,
            tags={"bench", "agent_forge"},
        ),
        memory.remember(
            "Ignition trace benchmark in agent forge reduced response latency and improved stability.",
            tier=MemoryTier.LONG_TERM,
            namespace=MemoryNamespace.KNOWLEDGE,
            memory_type=MemoryType.EPISODIC,
            tags={"bench", "agent_forge"},
        ),
        memory.remember(
            "Latency reduction from ignition trace benchmark yielded stable agent_forge execution.",
            tier=MemoryTier.LONG_TERM,
            namespace=MemoryNamespace.KNOWLEDGE,
            memory_type=MemoryType.EPISODIC,
            tags={"bench", "agent_forge"},
        ),
    ]
    for source_id in source_ids:
        _age_item(memory, source_id, days=90)

    report = memory.semantic_compress_old_memories(older_than_days=30, similarity_threshold=0.2, min_cluster_size=3)

    assert report["clusters_created"] == 1
    assert report["summaries_created"] == 1
    compressed = [item for item in memory.tiers[MemoryTier.LONG_TERM].values() if item.metadata.get("is_compressed")]
    assert len(compressed) == 1
    summary = compressed[0]
    assert summary.memory_type == MemoryType.SEMANTIC
    assert sorted(summary.metadata["source_ids"]) == sorted(source_ids)
    for source_id in source_ids:
        source = memory.tiers[MemoryTier.LONG_TERM][source_id]
        assert source.metadata["compressed_into"] == summary.id
        assert "compressed_source" in source.tags


def test_semantic_compress_old_memories_dry_run_does_not_mutate(tmp_path) -> None:
    memory = TieredMemorySystem(persistence_dir=tmp_path)
    ids = [
        memory.remember(
            "Kernel perturbation trial remained stable with low boundary error.", tier=MemoryTier.LONG_TERM
        ),
        memory.remember(
            "Boundary error stayed low during kernel perturbation benchmark trial.", tier=MemoryTier.LONG_TERM
        ),
        memory.remember(
            "Perturbation benchmark showed stable kernel behavior and low boundary drift.", tier=MemoryTier.LONG_TERM
        ),
    ]
    for memory_id in ids:
        _age_item(memory, memory_id, days=120)

    report = memory.semantic_compress_old_memories(
        older_than_days=30,
        similarity_threshold=0.2,
        min_cluster_size=3,
        dry_run=True,
    )

    assert report["potential_clusters"] == 1
    assert report["summaries_created"] == 0
    assert not [item for item in memory.tiers[MemoryTier.LONG_TERM].values() if item.metadata.get("is_compressed")]
    for memory_id in ids:
        assert "compressed_into" not in memory.tiers[MemoryTier.LONG_TERM][memory_id].metadata
