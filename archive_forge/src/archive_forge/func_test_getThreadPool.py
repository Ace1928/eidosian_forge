import gc
import threading
from weakref import ref
from twisted.internet.interfaces import IReactorThreads
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.python.threadable import isInIOThread
from twisted.python.threadpool import ThreadPool
from twisted.python.versions import Version
def test_getThreadPool(self):
    """
        C{reactor.getThreadPool()} returns an instance of L{ThreadPool} which
        starts when C{reactor.run()} is called and stops before it returns.
        """
    state = []
    reactor = self.buildReactor()
    pool = reactor.getThreadPool()
    self.assertIsInstance(pool, ThreadPool)
    self.assertFalse(pool.started, 'Pool should not start before reactor.run')

    def f():
        state.append(pool.started)
        state.append(pool.joined)
        reactor.stop()
    reactor.callWhenRunning(f)
    self.runReactor(reactor, 2)
    self.assertTrue(state[0], 'Pool should start after reactor.run')
    self.assertFalse(state[1], 'Pool should not be joined before reactor.stop')
    self.assertTrue(pool.joined, 'Pool should be stopped after reactor.run returns')