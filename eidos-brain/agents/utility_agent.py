"""General-purpose utilities for Eidos."""

from .base_agent import Agent


class UtilityAgent(Agent):
    """Provides supporting functions for the system."""

    def act(self, task: str) -> str:
        """Perform a single utility task and return a status message."""
        return f"Performed {task}"

    def batch_perform(self, tasks: list[str]) -> list[str]:
        """Perform multiple tasks and collect status messages."""
        return self.act_all(tasks)
