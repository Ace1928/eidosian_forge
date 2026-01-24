"""Worker management utilities for Word Forge."""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Protocol
from eidosian_core import eidosian


class Worker(Protocol):
    """Minimal protocol all workers must implement."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: Optional[float] = None) -> None: ...


class WorkerManager:
    """Register and control background workers."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._workers: List[Worker] = []

    @eidosian()
    def register(self, worker: Worker) -> None:
        """Add a worker to be managed."""
        self._workers.append(worker)
        self.logger.debug("Registered worker %s", worker)

    @eidosian()
    def start_all(self) -> None:
        """Start all registered workers."""
        for w in self._workers:
            try:
                w.start()
                self.logger.debug("Started worker %s", w)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("Failed to start %s: %s", w, exc)

    @eidosian()
    def stop_all(self, join: bool = True, timeout: float = 5.0) -> None:
        """Stop all workers and optionally join their threads."""
        for w in self._workers:
            try:
                w.stop()
                self.logger.debug("Stopped worker %s", w)
            except Exception as exc:  # pragma: no cover - defensive
                self.logger.error("Failed to stop %s: %s", w, exc)

        if join:
            for w in self._workers:
                if hasattr(w, "join"):
                    try:
                        w.join(timeout=timeout)
                    except Exception:
                        pass

    @eidosian()
    def any_alive(self) -> bool:
        """Check if any managed worker is still running."""
        alive = False
        for w in self._workers:
            try:
                if hasattr(w, "is_alive") and w.is_alive():
                    alive = True
                    break
            except Exception:
                continue
        return alive

    def __iter__(self) -> Iterable[Worker]:
        return iter(self._workers)
