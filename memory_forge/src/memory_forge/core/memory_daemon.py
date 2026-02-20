"""
Memory Retrieval Daemon: monitors prompts/commands/output logs and suggests memories.

This is a lightweight, file-based daemon intended to be run in the background.
It watches a log file and writes ranked memory suggestions to a sidecar JSON file.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from eidosian_core import eidosian

from .config import MemoryConfig
from .main import MemoryForge
from .memory_broker import MemoryBroker
from .memory_retrieval import MemoryRetrievalEngine, RetrievalResult


@dataclass
class DaemonConfig:
    log_path: Path
    output_path: Path
    poll_interval: float = 1.5
    max_results: int = 5


class MemoryRetrievalDaemon:
    def __init__(self, data_dir: Path, config: DaemonConfig):
        forge_config = MemoryConfig()
        forge_config.episodic.type = "json"
        forge_config.episodic.connection_string = str(data_dir / "episodic_memory.json")
        broker = MemoryBroker(data_dir=data_dir, forge=MemoryForge(forge_config))
        self.engine = MemoryRetrievalEngine(broker)
        self.config = config
        self._last_size = 0

    def _read_new_lines(self) -> str:
        if not self.config.log_path.exists():
            return ""
        data = self.config.log_path.read_text(encoding="utf-8")
        if len(data) <= self._last_size:
            return ""
        new = data[self._last_size :]
        self._last_size = len(data)
        return new.strip()

    def _write_output(self, results: List[RetrievalResult]) -> None:
        payload = [
            {
                "source": r.source,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @eidosian()
    def run_forever(self) -> None:
        while True:
            new_text = self._read_new_lines()
            if new_text:
                suggestions = self.engine.suggest(new_text, limit=self.config.max_results)
                self._write_output(suggestions)
            time.sleep(self.config.poll_interval)
