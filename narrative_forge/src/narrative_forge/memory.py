import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Deque
from collections import deque


@dataclass
class Memory:
    """Persistent memory structure for interactions, events and glossary."""

    interactions: Deque[Dict[str, Any]] = field(
        default_factory=lambda: deque(maxlen=1000)
    )
    events: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=1000))
    glossary: Dict[str, int] = field(default_factory=dict)


class MemoryStore:
    """Handles loading and saving Memory to a JSON file."""

    def __init__(self, path: str = "memory.json") -> None:
        self.path = path
        self.data = Memory()
        self.load()

    def load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self.data = Memory()
            self.data.interactions.extend(raw.get("interactions", []))
            self.data.events.extend(raw.get("events", []))
            self.data.glossary.update(raw.get("glossary", {}))
        else:
            self.data = Memory()

    def save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "interactions": list(self.data.interactions),
                    "events": list(self.data.events),
                    "glossary": self.data.glossary,
                },
                fh,
                indent=2,
            )

