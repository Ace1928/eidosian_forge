from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import (
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase
def test_resolves13RootServers(self):
    """
        The L{IResolverSimple} supplied to L{root.bootstrap} is used to lookup
        the IP addresses of the 13 root name servers.
        """
    stubResolver = StubResolver()
    root.bootstrap(stubResolver)
    self.assertEqual(stubResolver.calls, [((s,), {}) for s in ROOT_SERVERS])