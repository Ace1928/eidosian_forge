import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertEqualMessage(self):
    """
        When a message is passed to L{assertEqual} it is included in the error
        message.
        """
    message = 'message'
    exception = self.assertRaises(self.failureException, self.assertEqual, 'foo', 'bar', message)
    self.assertIn(message, str(exception))