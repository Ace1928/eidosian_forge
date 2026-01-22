import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessRaises_expected(self):
    x = self.failUnlessRaises(ValueError, self._raiseError, ValueError)
    self.assertTrue(isinstance(x, ValueError), 'Expect failUnlessRaises to return instance of raised exception.')