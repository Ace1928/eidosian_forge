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
def test_missingName(self):
    """
        If a name is missing, L{Resolver.lookupAddress} returns a L{Deferred}
        which fails with L{DNSNameError}.
        """
    servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'rCode': ENAME}}}
    resolver = self._getResolver(servers)
    d = resolver.lookupAddress(b'foo.example.com')
    return self.assertFailure(d, DNSNameError)