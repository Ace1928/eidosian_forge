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
def test_reactor(self):
    """
        The L{client.Resolver} included in the L{resolve.ResolverChain} returned
        by L{client.createResolver} uses the global reactor.
        """
    reactor = Clock()
    with AlternateReactor(reactor):
        resolver = client.createResolver()
    res = [r for r in resolver.resolvers if isinstance(r, client.Resolver)]
    self.assertEqual(1, len(res))
    self.assertIs(reactor, res[0]._reactor)