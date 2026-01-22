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
def test_searchEmptyArgument(self):
    """
        L{client.Resolver.parseConfig} treats a I{search} line without an
        argument as indicating an empty search suffix.
        """
    resolver = client.Resolver(servers=[('127.0.0.1', 53)])
    resolver.parseConfig([b'search\n'])
    self.assertEqual([], resolver.search)