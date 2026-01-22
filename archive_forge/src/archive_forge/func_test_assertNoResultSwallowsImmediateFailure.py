import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertNoResultSwallowsImmediateFailure(self):
    """
        When passed a L{Deferred} which currently has a L{Failure} result,
        L{SynchronousTestCase.assertNoResult} changes the result of the
        L{Deferred} to a success.
        """
    d = fail(self.failure)

    async def raisesException():
        return await d
    c = raisesException()
    try:
        self.assertNoResult(d)
    except self.failureException:
        pass
    self.assertEqual(None, self.successResultOf(c))