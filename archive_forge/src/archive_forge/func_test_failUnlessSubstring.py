import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessSubstring(self):
    x = 'cat'
    y = 'the dog sat'
    z = 'the cat sat'
    self.failUnlessSubstring(x, x)
    ret = self.failUnlessSubstring(x, z)
    self.assertEqual(ret, x, 'should return first parameter')
    self.failUnlessRaises(self.failureException, self.failUnlessSubstring, x, y)
    self.failUnlessRaises(self.failureException, self.failUnlessSubstring, z, x)