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
def test_never_again_exc(self):
    m_1 = mock.MagicMock()
    m_2 = mock.MagicMock()

    @periodics.periodic(0.5)
    def run_only_once():
        m_1()
        raise periodics.NeverAgain('No need to run again !!')

    @periodics.periodic(0.5)
    def keep_running():
        m_2()
    callables = [(run_only_once, None, None), (keep_running, None, None)]
    executor_factory = lambda: self.executor_cls(**self.executor_kwargs)
    w = periodics.PeriodicWorker(callables, executor_factory=executor_factory, **self.worker_kwargs)
    with self.create_destroy(w.start):
        self.sleep(2.0)
        w.stop()
    for watcher in w.iter_watchers():
        self.assertGreaterEqual(watcher.runs, 1)
        self.assertGreaterEqual(watcher.successes, 1)
        self.assertEqual(watcher.failures, 0)
    self.assertEqual(m_1.call_count, 1)
    self.assertGreaterEqual(m_2.call_count, 3)