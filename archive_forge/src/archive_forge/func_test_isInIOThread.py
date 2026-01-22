import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_isInIOThread(self):
    """
        The reactor registers itself as the I/O thread when it runs so that
        L{twisted.python.threadable.isInIOThread} returns C{True} if it is
        called in the thread the reactor is running in.
        """
    results = []
    reactor = self.buildReactor()

    def check():
        results.append(isInIOThread())
        reactor.stop()
    reactor.callWhenRunning(check)
    self.runReactor(reactor)
    self.assertEqual([True], results)