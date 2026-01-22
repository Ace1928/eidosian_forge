import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithWrongExceptionOneExpectedExceptionHasTB(self):
    """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a coroutine
        that raises an exception with a failure type that was not expected, and
        the L{SynchronousTestCase.failureException} message contains the
        original exception traceback.
        """
    exception = Exception('Bad times')
    try:
        self.failureResultOf(self.raisesException(exception), KeyError)
    except self.failureException as e:
        self.assertIn('Failure of type (builtins.KeyError) expected on', str(e))
        self.assertIn('builtins.Exception: Bad times', str(e))