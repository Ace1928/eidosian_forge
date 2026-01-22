from collections import defaultdict
from socket import (
from threading import Lock, local
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted._threads import LockWorker, Team, createMemoryWorker
from twisted.internet._resolver import (
from twisted.internet.address import IPv4Address, IPv6Address
from twisted.internet.base import PluggableResolverMixin, ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.error import DNSLookupError
from twisted.internet.interfaces import (
from twisted.python.threadpool import ThreadPool
from twisted.trial.unittest import SynchronousTestCase as UnitTest
def test_logErrorsInThreads(self):
    """
        L{DeterministicThreadPool} will log any exceptions that its "thread"
        workers encounter.
        """
    self.pool, self.doThreadWork = deterministicPool()

    def divideByZero():
        return 1 / 0
    self.pool.callInThread(divideByZero)
    self.doThreadWork()
    self.assertEqual(len(self.flushLoggedErrors(ZeroDivisionError)), 1)