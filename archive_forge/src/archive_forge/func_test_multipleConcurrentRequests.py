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
def test_multipleConcurrentRequests(self):
    """
        L{client.Resolver.query} issues a request for each different concurrent
        query.
        """
    protocol = StubDNSDatagramProtocol()
    resolver = client.Resolver(servers=[('example.com', 53)])
    resolver._connectedProtocol = lambda: protocol
    queries = protocol.queries
    firstQuery = dns.Query(b'foo.example.com', dns.A)
    resolver.query(firstQuery)
    self.assertEqual(len(queries), 1)
    secondQuery = dns.Query(b'bar.example.com', dns.A)
    resolver.query(secondQuery)
    self.assertEqual(len(queries), 2)
    thirdQuery = dns.Query(b'foo.example.com', dns.A6)
    resolver.query(thirdQuery)
    self.assertEqual(len(queries), 3)