"""Agent dedicated to running experiments within Eidos-Brain."""

from .base_agent import Agent


class ExperimentAgent(Agent):
    """Handles experimental cycles and evaluations."""

    def act(self, hypothesis: str) -> str:
        """Execute an experiment and return its result."""
        return f"Experimenting with {hypothesis}"

    def run(self, hypothesis: str) -> str:
        """Backward-compatible wrapper for :meth:`act`."""
        import warnings

        warnings.warn(
            "The `run` method is deprecated and will be removed in a future release. "
            "Please use the `act` method directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.act(hypothesis)

    def run_series(self, hypotheses: list[str]) -> list[str]:
        """Run a series of experiments and collect results."""
        return self.act_all(hypotheses)
