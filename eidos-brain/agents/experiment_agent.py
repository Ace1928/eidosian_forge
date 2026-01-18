"""Agent dedicated to running experiments within Eidos-Brain."""

from .base_agent import Agent


class ExperimentAgent(Agent):
    """Handles experimental cycles and evaluations."""

    def act(self, hypothesis: str) -> str:
        """Execute an experiment and return its result."""
        return f"Experimenting with {hypothesis}"

    def run_series(self, hypotheses: list[str]) -> list[str]:
        """Run a series of experiments and collect results."""
        return self.act_all(hypotheses)
