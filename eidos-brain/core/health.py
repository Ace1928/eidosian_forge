"""Health-check utilities for container monitoring."""

from __future__ import annotations


class HealthChecker:
    """Provide a basic status report for the system."""

    def check(self) -> dict[str, str]:
        """Return a simple health dictionary."""
        return {"status": "ok"}
