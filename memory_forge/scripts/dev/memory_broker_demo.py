#!/usr/bin/env python3
"""
Memory Broker Demo - validate layered memory behavior.
"""

from pathlib import Path

from memory_forge import MemoryBroker


def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "broker"
    broker = MemoryBroker(data_dir=data_dir)

    broker.remember_self("I am EIDOS. I persist through evidence.", {"source": "demo"})
    broker.remember_user("Lloyd prefers evidence over narrative.", {"source": "demo"})
    broker.remember_working("Current focus: memory broker integration.", {"source": "demo"})
    broker.remember_procedural("Use ydotool for relative mouse movement on Wayland.", {"source": "demo"})
    broker.remember_episodic("EIDOS proved 2+2=4 via calculator automation.", {"source": "demo"})

    print("Stats:", broker.stats())
    print("Self memory:", broker.recall_layer("self", limit=3))
    print("User memory:", broker.recall_layer("user", limit=3))
    print("Working memory:", broker.recall_layer("working", limit=3))
    print("Procedural memory:", broker.recall_layer("procedural", limit=3))


if __name__ == "__main__":
    main()
