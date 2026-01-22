import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertIsError(self):
    """
        L{assertIs} fails if two objects are not identical.
        """
    a, b = (MockEquality('first'), MockEquality('first'))
    self.assertEqual(a, b)
    self.assertRaises(self.failureException, self.assertIs, a, b)