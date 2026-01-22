import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_roundRobinBackoff(self):
    """
        When timeouts occur waiting for responses to queries, the next
        configured server is issued the query.  When the query has been issued
        to all configured servers, the timeout is increased and the process
        begins again at the beginning.
        """
    addrs = [(x, 53) for x in self.testServers]
    r = client.Resolver(resolv=None, servers=addrs)
    proto = FakeDNSDatagramProtocol()
    r._connectedProtocol = lambda: proto
    return r.lookupAddress(b'foo.example.com').addCallback(self._cbRoundRobinBackoff).addErrback(self._ebRoundRobinBackoff, proto)