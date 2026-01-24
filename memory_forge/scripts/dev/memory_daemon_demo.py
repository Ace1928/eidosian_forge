#!/usr/bin/env python3
"""
Memory Daemon Demo - writes suggestions for a mock log file.
"""
from pathlib import Path
import time
from memory_forge import MemoryRetrievalDaemon, DaemonConfig, MemoryBroker


def main() -> None:
    base = Path(__file__).resolve().parents[2]
    data_dir = base / "data" / "broker"
    log_path = base / "data" / "memory_stream.log"
    output_path = base / "data" / "memory_suggestions.json"

    # Seed some memories
    broker = MemoryBroker(data_dir=data_dir)
    broker.remember_self("I am EIDOS. I prefer evidence over narrative.")
    broker.remember_user("Lloyd wants rigorous verification.")
    broker.remember_procedural("Use ydotool relative movement on Wayland.")
    broker.remember_episodic("Calculator automation proof exists.")

    # Write a sample log line
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("Need to verify mouse movement and OCR reliability.\n", encoding="utf-8")

    daemon = MemoryRetrievalDaemon(data_dir=data_dir, config=DaemonConfig(log_path=log_path, output_path=output_path))
    daemon.run_forever()


if __name__ == "__main__":
    main()
