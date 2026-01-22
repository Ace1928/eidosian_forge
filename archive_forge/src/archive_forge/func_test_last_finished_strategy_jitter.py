import contextlib
import functools
import threading
import time
from unittest import mock
import eventlet
from eventlet.green import threading as green_threading
import testscenarios
import futurist
from futurist import periodics
from futurist.tests import base
def test_last_finished_strategy_jitter(self):
    ev = self.event_cls()
    calls = []
    stop_after_calls = 3

    @periodics.periodic(0.1, run_immediately=False)
    def fast_periodic():
        calls.append(True)
        if len(calls) > stop_after_calls:
            ev.set()
    worker_kwargs = self.worker_kwargs.copy()
    worker_kwargs['schedule_strategy'] = 'last_finished_jitter'
    callables = [(fast_periodic, None, None)]
    w = periodics.PeriodicWorker(callables, **worker_kwargs)
    with self.create_destroy(w.start):
        ev.wait()
        w.stop()
    self.assertGreaterEqual(len(calls), stop_after_calls)