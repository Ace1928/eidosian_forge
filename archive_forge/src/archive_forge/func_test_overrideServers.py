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
def test_overrideServers(self):
    """
        Servers passed to L{client.createResolver} are used in addition to any
        found in the file given by the I{resolvconf} parameter.
        """
    resolvconf = self.path()
    resolvconf.setContent(b'nameserver 127.1.2.3\n')
    with AlternateReactor(Clock()):
        resolver = client.createResolver(servers=[('127.3.2.1', 53)], resolvconf=resolvconf.path)
    res = [r for r in resolver.resolvers if isinstance(r, client.Resolver)]
    self.assertEqual(1, len(res))
    self.assertEqual([('127.3.2.1', 53)], res[0].servers)
    self.assertEqual([('127.1.2.3', 53)], res[0].dynServers)