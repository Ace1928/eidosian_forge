import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
def test_callbackContext(self):
    """
        The context L{ThreadPool.callInThreadWithCallback} is invoked in is
        shared by the context the callable and C{onResult} callback are
        invoked in.
        """
    myctx = context.theContextTracker.currentContext().contexts[-1]
    myctx['testing'] = 'this must be present'
    contexts = []
    event = threading.Event()

    def onResult(success, result):
        ctx = context.theContextTracker.currentContext().contexts[-1]
        contexts.append(ctx)
        event.set()

    def func():
        ctx = context.theContextTracker.currentContext().contexts[-1]
        contexts.append(ctx)
    tp = threadpool.ThreadPool(0, 1)
    tp.callInThreadWithCallback(onResult, func)
    tp.start()
    self.addCleanup(tp.stop)
    event.wait(self.getTimeout())
    self.assertEqual(len(contexts), 2)
    self.assertEqual(myctx, contexts[0])
    self.assertEqual(myctx, contexts[1])