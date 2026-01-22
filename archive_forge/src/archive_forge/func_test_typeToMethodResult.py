from zope.interface.verify import verifyClass
from twisted.internet.interfaces import IResolver
from twisted.names.common import ResolverBase
from twisted.names.dns import EFORMAT, ENAME, ENOTIMP, EREFUSED, ESERVER, Query
from twisted.names.error import (
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase
def test_typeToMethodResult(self):
    """
        L{ResolverBase.query} returns a L{Deferred} which fires with the result
        of the method found in the C{typeToMethod} mapping for the type of the
        query passed to it.
        """
    expected = object()
    resolver = ResolverBase()
    resolver.typeToMethod = {54321: lambda query, timeout: expected}
    query = Query(name=b'example.com', type=54321)
    queryDeferred = resolver.query(query, 123)
    result = []
    queryDeferred.addBoth(result.append)
    self.assertEqual(expected, result[0])