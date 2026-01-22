import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessApproximates(self):
    x, y, z = (1.0, 1.1, 1.2)
    self.failUnlessApproximates(x, x, 0.2)
    ret = self.failUnlessApproximates(x, y, 0.2)
    self.assertEqual(ret, x, 'failUnlessApproximates should return first parameter')
    self.failUnlessRaises(self.failureException, self.failUnlessApproximates, x, z, 0.1)
    self.failUnlessRaises(self.failureException, self.failUnlessApproximates, x, y, 0.1)