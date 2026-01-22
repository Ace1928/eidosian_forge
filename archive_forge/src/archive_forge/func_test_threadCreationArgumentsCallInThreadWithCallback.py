import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_threadCreationArgumentsCallInThreadWithCallback(self):
    """
        As C{test_threadCreationArguments} above, but for
        callInThreadWithCallback.
        """
    tp = threadpool.ThreadPool(0, 1)
    tp.start()
    self.addCleanup(tp.stop)
    self.assertEqual(tp.threads, [])
    refdict = {}
    onResultWait = threading.Event()
    onResultDone = threading.Event()
    resultRef = []

    def onResult(success, result):
        gc.collect()
        onResultWait.wait(self.getTimeout())
        refdict['workerRef'] = workerRef()
        refdict['uniqueRef'] = uniqueRef()
        onResultDone.set()
        resultRef.append(weakref.ref(result))

    def worker(arg, test):
        return Dumb()

    class Dumb:
        pass
    unique = Dumb()
    onResultRef = weakref.ref(onResult)
    workerRef = weakref.ref(worker)
    uniqueRef = weakref.ref(unique)
    tp.callInThreadWithCallback(onResult, worker, unique, test=unique)
    del worker
    del unique
    onResultWait.set()
    onResultDone.wait(self.getTimeout())
    gc.collect()
    self.assertIsNone(uniqueRef())
    self.assertIsNone(workerRef())
    del onResult
    gc.collect()
    self.assertIsNone(onResultRef())
    self.assertIsNone(resultRef[0]())
    self.assertEqual(list(refdict.values()), [None, None])