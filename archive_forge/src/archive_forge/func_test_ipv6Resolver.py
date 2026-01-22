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
def test_ipv6Resolver(self):
    """
        If the resolver is ipv6, open a ipv6 port.
        """
    fake = test_util.MemoryReactor()
    resolver = client.Resolver(servers=[('::1', 53)], reactor=fake)
    resolver.query(dns.Query(b'foo.example.com'))
    [(proto, transport)] = fake.udpPorts.items()
    interface = transport.getHost().host
    self.assertEqual('::', interface)