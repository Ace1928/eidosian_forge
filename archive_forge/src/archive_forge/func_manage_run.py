import asyncio
import asyncio.events as events
import os
import sys
import threading
from contextlib import contextmanager, suppress
from heapq import heappop
@contextmanager
def manage_run(self):
    """Set up the loop for running."""
    self._check_closed()
    old_thread_id = self._thread_id
    old_running_loop = events._get_running_loop()
    try:
        self._thread_id = threading.get_ident()
        events._set_running_loop(self)
        self._num_runs_pending += 1
        if self._is_proactorloop:
            if self._self_reading_future is None:
                self.call_soon(self._loop_self_reading)
        yield
    finally:
        self._thread_id = old_thread_id
        events._set_running_loop(old_running_loop)
        self._num_runs_pending -= 1
        if self._is_proactorloop:
            if self._num_runs_pending == 0 and self._self_reading_future is not None:
                ov = self._self_reading_future._ov
                self._self_reading_future.cancel()
                if ov is not None:
                    self._proactor._unregister(ov)
                self._self_reading_future = None