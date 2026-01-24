#!/usr/bin/env python3
"""
Memory Retrieval Demo - validates suggestion pipeline.
"""
from pathlib import Path
from memory_forge import MemoryBroker, MemoryRetrievalEngine


def main() -> None:
    data_dir = Path(__file__).resolve().parents[2] / "data" / "broker"
    broker = MemoryBroker(data_dir=data_dir)

    broker.remember_self("I am EIDOS. I prioritize evidence over narrative.", {"tag": "identity"})
    broker.remember_user("Lloyd values rigorous verification.", {"tag": "user"})
    broker.remember_procedural("Use ydotool relative movements on Wayland.", {"tag": "procedure"})
    broker.remember_episodic("EIDOS verified calculator automation 2+2=4.")

    engine = MemoryRetrievalEngine(broker)
    suggestions = engine.suggest("verification evidence", limit=5)
    for s in suggestions:
        print(f"[{s.source}] {s.content}")


if __name__ == "__main__":
    main()
