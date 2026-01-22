import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_assertIsInstanceMultipleClasses(self):
    """
        Test a true condition of assertIsInstance with multiple classes.
        """
    A = type('A', (object,), {})
    B = type('B', (object,), {})
    a = A()
    self.assertIsInstance(a, (A, B))