import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_suggestThreadPoolSize(self):
    """
        C{reactor.suggestThreadPoolSize()} sets the maximum size of the reactor
        threadpool.
        """
    reactor = self.buildReactor()
    reactor.suggestThreadPoolSize(17)
    pool = reactor.getThreadPool()
    self.assertEqual(pool.max, 17)