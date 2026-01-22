import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_workerStateTransition(self):
    """
        As the worker receives and completes work, it transitions between
        the working and waiting states.
        """
    pool = threadpool.ThreadPool(0, 1)
    pool.start()
    self.addCleanup(pool.stop)
    self.assertEqual(pool.workers, 0)
    self.assertEqual(len(pool.waiters), 0)
    self.assertEqual(len(pool.working), 0)
    threadWorking = threading.Event()
    threadFinish = threading.Event()

    def _thread():
        threadWorking.set()
        threadFinish.wait(10)
    pool.callInThread(_thread)
    threadWorking.wait(10)
    self.assertEqual(pool.workers, 1)
    self.assertEqual(len(pool.waiters), 0)
    self.assertEqual(len(pool.working), 1)
    threadFinish.set()
    while not len(pool.waiters):
        time.sleep(0.0005)
    self.assertEqual(len(pool.waiters), 1)
    self.assertEqual(len(pool.working), 0)