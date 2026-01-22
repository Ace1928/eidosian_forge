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
def test_portNumber(self):
    """
        L{SimpleResolverComplexifier} preserves the C{port} argument passed to
        C{resolveHostName} in its returned addresses.
        """
    simple = SillyResolverSimple()
    complex = SimpleResolverComplexifier(simple)
    receiver = ResultHolder(self)
    complex.resolveHostName(receiver, 'example.com', 4321)
    self.assertEqual(receiver._started, True)
    self.assertEqual(receiver._ended, False)
    self.assertEqual(receiver._addresses, [])
    simple._requests[0].callback('192.168.1.1')
    self.assertEqual(receiver._addresses, [IPv4Address('TCP', '192.168.1.1', 4321)])
    self.assertEqual(receiver._ended, True)