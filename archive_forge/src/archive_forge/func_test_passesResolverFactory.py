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
def test_passesResolverFactory(self):
    """
        L{root.bootstrap} accepts a C{resolverFactory} argument which is passed
        as an argument to L{root.Resolver} when it has successfully looked up
        root hints.
        """
    stubResolver = StubResolver()
    deferredResolver = root.bootstrap(stubResolver, resolverFactory=raisingResolverFactory)
    for d in stubResolver.pendingResults:
        d.callback('192.0.2.101')
    self.assertIs(deferredResolver._resolverFactory, raisingResolverFactory)