import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_successResultOfWithSuccessResult(self):
    """
        When passed a coroutine which currently has a result (ie, if converted
        into a L{Deferred}, L{Deferred.addCallback} would cause the added
        callback to be called before C{addCallback} returns),
        L{SynchronousTestCase.successResultOf} returns that result.
        """
    self.assertIdentical(self.result, self.successResultOf(self.successResult()))