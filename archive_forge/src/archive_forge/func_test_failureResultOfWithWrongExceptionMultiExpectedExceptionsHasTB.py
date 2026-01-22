import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failureResultOfWithWrongExceptionMultiExpectedExceptionsHasTB(self):
    """
        L{SynchronousTestCase.failureResultOf} raises
        L{SynchronousTestCase.failureException} when called with a coroutine
        that raises an exception of a type that was not expected, and the
        L{SynchronousTestCase.failureException} message contains the original
        exception traceback in the error message.
        """
    exception = Exception('Bad times')
    try:
        self.failureResultOf(self.raisesException(exception), KeyError, IOError)
    except self.failureException as e:
        self.assertIn('Failure of type (builtins.KeyError or builtins.OSError) expected on', str(e))
        self.assertIn('builtins.Exception: Bad times', str(e))