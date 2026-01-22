import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_callInThreadWithCallback(self):
    """
        L{ThreadPool.callInThreadWithCallback} calls C{onResult} with a
        two-tuple of C{(True, result)} where C{result} is the value returned
        by the callable supplied.
        """
    waiter = threading.Lock()
    waiter.acquire()
    results = []

    def onResult(success, result):
        waiter.release()
        results.append(success)
        results.append(result)
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThreadWithCallback(onResult, lambda: 'test')
    tp.start()
    try:
        self._waitForLock(waiter)
    finally:
        tp.stop()
    self.assertTrue(results[0])
    self.assertEqual(results[1], 'test')