import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithFailureResult(self):
    """
        When passed a L{Deferred} which currently has a L{Failure} result
        (ie, L{Deferred.addErrback} would cause the added errback to be called
        before C{addErrback} returns), L{SynchronousTestCase.failureResultOf}
        returns that L{Failure}.
        """
    self.assertIdentical(self.failure, self.failureResultOf(fail(self.failure)))