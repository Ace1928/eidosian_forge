import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failIfSubstring(self):
    x = 'cat'
    y = 'the dog sat'
    z = 'the cat sat'
    self.failIfSubstring(z, x)
    ret = self.failIfSubstring(x, y)
    self.assertEqual(ret, x, 'should return first parameter')
    self.failUnlessRaises(self.failureException, self.failIfSubstring, x, x)
    self.failUnlessRaises(self.failureException, self.failIfSubstring, x, z)