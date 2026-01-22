import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_stopThreadPoolWhenStartedAfterReactorRan(self):
    """
        We must handle the case of shutting down the thread pool when it was
        started after the reactor was run in a special way.

        Some implementation background: The thread pool is started with
        callWhenRunning, which only returns a system trigger ID when it is
        invoked before the reactor is started.

        This is the case of the thread pool being created after the reactor
        is started.
        """
    reactor = self.buildReactor()
    threadPoolRefs = []

    def acquireThreadPool():
        threadPoolRefs.append(ref(reactor.getThreadPool()))
        reactor.stop()
    reactor.callWhenRunning(acquireThreadPool)
    self.runReactor(reactor)
    gc.collect()
    self.assertIsNone(threadPoolRefs[0]())