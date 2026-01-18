"""Foundational agent interface used across Eidos-Brain."""

from __future__ import annotations

from abc import ABC, abstractmethod


class Agent(ABC):
    """Provide standard hooks for all specialized agents.

    Subclasses must implement :meth:`act`. Batch operations are built on
    top of this single-action interface to encourage minimal, testable
    behaviors. The method names mirror the patterns documented in
    ``knowledge/templates.md`` and ``knowledge/recursive_patterns.md``.
    """

    @abstractmethod
    def act(self, item: str) -> str:
        """Handle a single item and return a status message."""
        raise NotImplementedError

    def act_all(self, items: list[str]) -> list[str]:
        """Run :meth:`act` across ``items`` and return aggregated results."""
        return [self.act(i) for i in items]
