import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.eidos_core import EidosCore


def test_memory_and_reflection():
    core = EidosCore()
    core.remember("hello")
    assert core.reflect() == ["hello"]


def test_recurse_adds_insights():
    core = EidosCore()
    core.remember("hello")
    core.recurse()
    memories = core.reflect()
    assert any(isinstance(m, dict) for m in memories)
    assert len(memories) == 2


def test_process_cycle_combines_steps():
    core = EidosCore()
    core.process_cycle("data")
    memories = core.reflect()
    assert "data" in memories
    assert any(isinstance(m, dict) and m.get("repr") == "'data'" for m in memories)
    assert len(memories) == 2
