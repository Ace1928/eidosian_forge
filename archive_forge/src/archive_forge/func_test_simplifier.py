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
def test_simplifier(self):
    """
        L{ComplexResolverSimplifier} translates an L{IHostnameResolver} into an
        L{IResolverSimple} for applications that still expect the old
        interfaces to be in place.
        """
    self.pool, self.doThreadWork = deterministicPool()
    self.reactor, self.doReactorWork = deterministicReactorThreads()
    self.getter = FakeAddrInfoGetter()
    self.resolver = GAIResolver(self.reactor, lambda: self.pool, self.getter.getaddrinfo)
    simpleResolver = ComplexResolverSimplifier(self.resolver)
    self.getter.addResultForHost('example.com', ('192.168.3.4', 4321))
    success = simpleResolver.getHostByName('example.com')
    failure = simpleResolver.getHostByName('nx.example.com')
    self.doThreadWork()
    self.doReactorWork()
    self.doThreadWork()
    self.doReactorWork()
    self.assertEqual(self.failureResultOf(failure).type, DNSLookupError)
    self.assertEqual(self.successResultOf(success), '192.168.3.4')