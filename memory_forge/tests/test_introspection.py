from __future__ import annotations

from datetime import datetime

from memory_forge.core.introspection import MemoryIntrospector
from memory_forge.core.tiered_memory import MemoryTier, TieredMemorySystem


def test_memory_introspector_normalizes_naive_created_at(tmp_path) -> None:
    memory = TieredMemorySystem(persistence_dir=tmp_path)
    memory_id = memory.remember("Naive timestamp memory", tier=MemoryTier.LONG_TERM)
    item = memory.tiers[MemoryTier.LONG_TERM][memory_id]
    item.created_at = datetime(2024, 1, 1, 12, 0, 0)
    item.last_accessed = datetime(2024, 1, 1, 12, 30, 0)
    memory._persist_tier(MemoryTier.LONG_TERM)

    stats = MemoryIntrospector(memory_dir=tmp_path).get_stats()

    assert stats.total_memories == 1
    assert stats.oldest_memory is not None
    assert stats.oldest_memory.tzinfo is not None
    assert stats.newest_memory is not None
    assert stats.newest_memory.tzinfo is not None


def test_memory_introspection_summary_uses_utc_timestamp(tmp_path) -> None:
    memory = TieredMemorySystem(persistence_dir=tmp_path)
    memory.remember("Summary timestamp check", tier=MemoryTier.LONG_TERM)

    summary = MemoryIntrospector(memory_dir=tmp_path).generate_summary()

    generated_line = next(line for line in summary.splitlines() if line.startswith("Generated: "))
    assert "+00:00" in generated_line or generated_line.endswith("Z")
