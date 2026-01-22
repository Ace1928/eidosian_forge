import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failIfIdenticalPositive(self):
    """
        C{failIfIdentical} returns its first argument if its first and second
        arguments are not the same object.
        """
    x = object()
    y = object()
    result = self.failIfIdentical(x, y)
    self.assertEqual(x, result)