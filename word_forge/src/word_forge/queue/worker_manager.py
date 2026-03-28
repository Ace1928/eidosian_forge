"""Worker management utilities for Word Forge."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Iterable, List, Optional, Protocol

from eidosian_core import eidosian
from word_forge.utils import metrics
from word_forge.utils.result import Result, failure, success


class WorkerState(Enum):
    """Execution states for background workers."""
    IDLE = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    FAILED = auto()


class Worker(Protocol):
    """Minimal protocol all workers must implement."""

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def is_alive(self) -> bool: ...

    def join(self, timeout: Optional[float] = None) -> None: ...

    @property
    def name(self) -> str: ...


@dataclass
class WorkerMetadata:
    """Metadata and statistics for a managed worker."""
    name: str
    start_time: float = 0.0
    restart_count: int = 0
    state: WorkerState = WorkerState.IDLE
    last_heartbeat: float = 0.0
    error_count: int = 0
    last_error: Optional[str] = None


class WorkerManager:
    """Register and control background workers with health monitoring."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self._workers: Dict[str, Worker] = {}
        self._metadata: Dict[str, WorkerMetadata] = {}
        self._max_restarts = 3

    @eidosian()
    def register(self, worker: Worker) -> None:
        """Add a worker to be managed."""
        name = getattr(worker, "name", str(worker))
        self._workers[name] = worker
        self._metadata[name] = WorkerMetadata(name=name)
        self.logger.debug("Registered worker %s", name)

    @eidosian()
    def start_all(self) -> None:
        """Start all registered workers."""
        for name, w in self._workers.items():
            self._start_worker(name)

    def _start_worker(self, name: str) -> bool:
        """Internal helper to start a single worker."""
        worker = self._workers[name]
        meta = self._metadata[name]
        try:
            meta.state = WorkerState.STARTING
            self._update_metrics(name, meta.state)
            worker.start()
            meta.state = WorkerState.RUNNING
            self._update_metrics(name, meta.state)
            meta.start_time = time.time()
            meta.last_heartbeat = time.time()
            self.logger.debug("Started worker %s", name)
            return True
        except Exception as exc:
            meta.state = WorkerState.FAILED
            self._update_metrics(name, meta.state)
            meta.error_count += 1
            meta.last_error = str(exc)
            self.logger.error("Failed to start %s: %s", name, exc)
            return False

    def _update_metrics(self, name: str, current_state: WorkerState) -> None:
        """Update Prometheus gauges for all possible states of a worker."""
        for state in WorkerState:
            value = 1.0 if state == current_state else 0.0
            metrics.set_worker_state(name, state.name, value)

    @eidosian()
    def any_alive(self) -> bool:
        """Return True when any managed worker currently reports alive."""
        return any(self._worker_is_alive(worker) for worker in self._workers.values())

    def _worker_is_alive(self, worker: Worker) -> bool:
        """Safely query worker liveness without propagating worker errors."""
        try:
            return bool(worker.is_alive())
        except Exception:
            return False

    @eidosian()
    def stop_all(self, join: bool = True, timeout: float = 5.0) -> None:
        """Stop all workers and optionally join their threads."""
        for name, w in self._workers.items():
            try:
                self._metadata[name].state = WorkerState.STOPPING
                self._update_metrics(name, self._metadata[name].state)
                w.stop()
                self._metadata[name].state = WorkerState.STOPPED
                self._update_metrics(name, self._metadata[name].state)
                self.logger.debug("Stopped worker %s", name)
            except Exception as exc:
                self._metadata[name].state = WorkerState.FAILED
                self._update_metrics(name, self._metadata[name].state)
                self.logger.error("Failed to stop %s: %s", name, exc)

        if join:
            for w in self._workers.values():
                if hasattr(w, "join"):
                    try:
                        w.join(timeout=timeout)
                    except Exception:
                        pass

    @eidosian()
    def monitor_health(self) -> None:
        """Check health of all workers and attempt restarts if policy allows."""
        for name, worker in self._workers.items():
            meta = self._metadata[name]
            
            # Check if alive
            is_alive = self._worker_is_alive(worker)

            if not is_alive and meta.state == WorkerState.RUNNING:
                self.logger.warning("Worker %s died unexpectedly", name)
                meta.state = WorkerState.FAILED
                self._update_metrics(name, meta.state)
                
                if meta.restart_count < self._max_restarts:
                    self.logger.info("Attempting restart %d/%d for %s", 
                                     meta.restart_count + 1, self._max_restarts, name)
                    meta.restart_count += 1
                    self._start_worker(name)
                else:
                    self.logger.error("Worker %s exceeded restart limit", name)
            elif is_alive and meta.state == WorkerState.RUNNING:
                # Periodic metric heartbeat
                self._update_metrics(name, meta.state)

    @eidosian()
    def get_stats(self) -> List[Dict[str, Any]]:
        """Retrieve statistics for all managed workers."""
        return [
            {
                "name": m.name,
                "state": m.state.name,
                "uptime": time.time() - m.start_time if m.state == WorkerState.RUNNING else 0,
                "restarts": m.restart_count,
                "errors": m.error_count,
                "last_error": m.last_error
            }
            for m in self._metadata.values()
        ]

    def __iter__(self) -> Iterable[Worker]:
        return iter(self._workers.values())
