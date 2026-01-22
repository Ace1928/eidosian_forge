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
def test_becomesResolver(self):
    """
        The L{root.DeferredResolver} initially returned by L{root.bootstrap}
        becomes a L{root.Resolver} when the supplied resolver has successfully
        looked up all root hints.
        """
    stubResolver = StubResolver()
    deferredResolver = root.bootstrap(stubResolver)
    for d in stubResolver.pendingResults:
        d.callback('192.0.2.101')
    self.assertIsInstance(deferredResolver, Resolver)