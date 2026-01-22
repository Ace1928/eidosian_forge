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
def test_pendingEmptiedInPlaceOnError(self):
    """
        When the TCP connection attempt fails, the
        L{client.Resolver.pending} list is emptied in place. It is not
        replaced with a new empty list.
        """
    reactor = proto_helpers.MemoryReactor()
    resolver = client.Resolver(servers=[('192.0.2.100', 53)], reactor=reactor)
    d = resolver.queryTCP(dns.Query('example.com'))
    host, port, factory, timeout, bindAddress = reactor.tcpClients[0]
    prePending = resolver.pending
    self.assertEqual(len(prePending), 1)

    class SentinelException(Exception):
        pass
    factory.clientConnectionFailed(reactor.connectors[0], failure.Failure(SentinelException()))
    self.failureResultOf(d, SentinelException)
    self.assertIs(resolver.pending, prePending)
    self.assertEqual(len(prePending), 0)