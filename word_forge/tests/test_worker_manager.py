"""Tests for word_forge.queue.worker_manager module.

This module tests the WorkerManager class which coordinates multiple workers.
"""

import threading
import time

from word_forge.queue.worker_manager import WorkerManager


class SimpleWorker(threading.Thread):
    """A simple worker thread for testing WorkerManager coordination."""

    def __init__(self):
        super().__init__(daemon=True)
        self._stop_event = threading.Event()
        self.started_flag = False

    def run(self):
        """Run the worker until stopped."""
        self.started_flag = True
        while not self._stop_event.is_set():
            time.sleep(0.01)

    def stop(self):
        """Signal the worker to stop."""
        self._stop_event.set()


def test_start_and_stop_all():
    """Test that WorkerManager can start and stop all registered workers."""
    w1 = SimpleWorker()
    w2 = SimpleWorker()
    manager = WorkerManager()
    manager.register(w1)
    manager.register(w2)

    manager.start_all()
    time.sleep(0.05)  # Give workers time to start
    assert w1.started_flag and w2.started_flag
    assert manager.any_alive()

    manager.stop_all()
    w1.join(timeout=1)
    w2.join(timeout=1)
    assert not manager.any_alive()


def test_register_workers():
    """Test that workers can be registered with the manager."""
    manager = WorkerManager()
    w1 = SimpleWorker()
    w2 = SimpleWorker()

    manager.register(w1)
    manager.register(w2)

    # Manager should track both workers
    assert len(manager._workers) == 2


def test_any_alive_with_no_workers():
    """Test any_alive returns False when no workers registered."""
    manager = WorkerManager()
    assert not manager.any_alive()


def test_stop_all_with_no_workers():
    """Test stop_all doesn't raise when no workers registered."""
    manager = WorkerManager()
    # Should not raise
    manager.stop_all()
