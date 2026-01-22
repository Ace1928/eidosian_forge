import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_successResultOfWithFailureHasTraceback(self):
    """
        L{SynchronousTestCase.successResultOf} raises a
        L{SynchronousTestCase.failureException} that has the original failure
        traceback when called with a coroutine with a failure result.
        """
    exception = Exception('Bad times')
    try:
        self.successResultOf(self.raisesException(exception))
    except self.failureException as e:
        self.assertIn('Success result expected on', str(e))
        self.assertIn('builtins.Exception: Bad times', str(e))