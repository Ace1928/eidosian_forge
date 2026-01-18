"""General-purpose utilities for Eidos."""

import warnings

from .base_agent import Agent


class UtilityAgent(Agent):
    """Provides supporting functions for the system."""

    def act(self, task: str) -> str:
        """Perform a single utility task and return a status message."""
        return f"Performed {task}"

    def perform_task(self, task: str) -> str:
        """Backward-compatible wrapper for :meth:`act`."""
        warnings.warn(
            "perform_task is deprecated and will be removed in a future release. "
            "Please use the act method instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.act(task)

    def batch_perform(self, tasks: list[str]) -> list[str]:
        """Perform multiple tasks and collect status messages."""
        return self.act_all(tasks)
