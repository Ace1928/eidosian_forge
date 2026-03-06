from memory_forge.core.interfaces import MemoryType
from memory_forge.core.tiered_memory import MemoryNamespace, MemoryTier, TieredMemorySystem


def test_remember_merges_duplicate_entries_across_instances(tmp_path) -> None:
    memory_a = TieredMemorySystem(persistence_dir=tmp_path)
    memory_b = TieredMemorySystem(persistence_dir=tmp_path)

    first_id = memory_a.remember(
        "Qwen3.5 thinking-off mode is the stable default for this Termux runtime.",
        tier=MemoryTier.LONG_TERM,
        namespace=MemoryNamespace.KNOWLEDGE,
        memory_type=MemoryType.SEMANTIC,
        tags={"qwen", "stable"},
        metadata={"model": "qwen3.5:2b", "mode": "off"},
    )
    second_id = memory_b.remember(
        "Qwen3.5 thinking-off mode is the stable default for this Termux runtime.",
        tier=MemoryTier.LONG_TERM,
        namespace=MemoryNamespace.KNOWLEDGE,
        memory_type=MemoryType.SEMANTIC,
        tags={"termux", "default"},
        metadata={"model": "qwen3.5:2b", "mode": "off"},
    )

    assert second_id == first_id

    reloaded = TieredMemorySystem(persistence_dir=tmp_path)
    persisted = list(reloaded.tiers[MemoryTier.LONG_TERM].values())
    assert len(persisted) == 1
    assert persisted[0].id == first_id
    assert persisted[0].tags == {"qwen", "stable", "termux", "default"}
