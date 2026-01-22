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
def test_add_with_auto_stop_when_empty_set(self):
    m = mock.Mock()

    @periodics.periodic(0.5)
    def run_only_once():
        raise periodics.NeverAgain('No need to run again !!')
    callables = [(run_only_once, None, None)]
    executor_factory = lambda: self.executor_cls(**self.executor_kwargs)
    w = periodics.PeriodicWorker(callables, executor_factory=executor_factory, **self.worker_kwargs)
    with self.create_destroy(w.start, auto_stop_when_empty=True):
        self.sleep(2.0)
        w.add(every_half_sec, m, None)
    m.assert_not_called()