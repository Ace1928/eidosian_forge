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
def test_disallowedPort(self):
    """
        If a port number is initially selected which cannot be bound, the
        L{CannotListenError} is handled and another port number is attempted.
        """
    ports = []

    class FakeReactor:

        def listenUDP(self, port, *args, **kwargs):
            ports.append(port)
            if len(ports) == 1:
                raise CannotListenError(None, port, None)
    resolver = client.Resolver(servers=[('example.com', 53)])
    resolver._reactor = FakeReactor()
    resolver._connectedProtocol()
    self.assertEqual(len(set(ports)), 2)