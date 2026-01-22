import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_callInThreadWithCallbackExceptionInOnResult(self):
    """
        L{ThreadPool.callInThreadWithCallback} logs the exception raised by
        C{onResult}.
        """

    class NewError(Exception):
        pass
    waiter = threading.Lock()
    waiter.acquire()
    results = []

    def onResult(success, result):
        results.append(success)
        results.append(result)
        raise NewError()
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThreadWithCallback(onResult, lambda: None)
    tp.callInThread(waiter.release)
    tp.start()
    try:
        self._waitForLock(waiter)
    finally:
        tp.stop()
    errors = self.flushLoggedErrors(NewError)
    self.assertEqual(len(errors), 1)
    self.assertTrue(results[0])
    self.assertIsNone(results[1])