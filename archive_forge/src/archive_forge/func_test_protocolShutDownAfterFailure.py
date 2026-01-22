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
def test_protocolShutDownAfterFailure(self):
    """
        If the L{Deferred} returned by L{DNSDatagramProtocol.query} fires with
        a failure, the L{DNSDatagramProtocol} is still disconnected from its
        transport.
        """

    class ExpectedException(Exception):
        pass
    resolver = client.Resolver(servers=[('example.com', 53)])
    protocols = []
    result = defer.Deferred()

    class FakeProtocol:

        def __init__(self):
            self.transport = StubPort()

        def query(self, address, query, timeout=10, id=None):
            protocols.append(self)
            return result
    resolver._connectedProtocol = FakeProtocol
    queryResult = resolver.query(dns.Query(b'foo.example.com'))
    self.assertFalse(protocols[0].transport.disconnected)
    result.errback(failure.Failure(ExpectedException()))
    self.assertTrue(protocols[0].transport.disconnected)
    return self.assertFailure(queryResult, ExpectedException)