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
def test_resolveOneHost(self):
    """
        Resolving an individual hostname that results in one address from
        getaddrinfo results in a single call each to C{resolutionBegan},
        C{addressResolved}, and C{resolutionComplete}.
        """
    receiver = ResultHolder(self)
    self.getter.addResultForHost('sample.example.com', ('4.3.2.1', 0))
    resolution = self.resolver.resolveHostName(receiver, 'sample.example.com')
    self.assertIs(receiver._resolution, resolution)
    self.assertEqual(receiver._started, True)
    self.assertEqual(receiver._ended, False)
    self.doThreadWork()
    self.doReactorWork()
    self.assertEqual(receiver._ended, True)
    self.assertEqual(receiver._addresses, [IPv4Address('TCP', '4.3.2.1', 0)])