from __future__ import annotations
import threading
import time
from contextlib import contextmanager
from queue import Queue
from typing import TYPE_CHECKING, Iterator, NoReturn
import pytest
from .. import _thread_cache
from .._thread_cache import ThreadCache, start_thread_soon
from .tutil import gc_collect_harder, slow
def test_race_between_idle_exit_and_job_assignment(monkeypatch: pytest.MonkeyPatch) -> None:

    class JankyLock:

        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._counter = 3

        def acquire(self, timeout: int=-1) -> bool:
            got_it = self._lock.acquire(timeout=timeout)
            if timeout == -1:
                return True
            elif got_it:
                if self._counter > 0:
                    self._counter -= 1
                    self._lock.release()
                    return False
                return True
            else:
                return False

        def release(self) -> None:
            self._lock.release()
    monkeypatch.setattr(_thread_cache, 'Lock', JankyLock)
    with _join_started_threads():
        tc = ThreadCache()
        done = threading.Event()
        tc.start_thread_soon(lambda: None, lambda _: done.set())
        done.wait()
        monkeypatch.setattr(_thread_cache, 'IDLE_TIMEOUT', 0.0001)
        tc.start_thread_soon(lambda: None, lambda _: None)