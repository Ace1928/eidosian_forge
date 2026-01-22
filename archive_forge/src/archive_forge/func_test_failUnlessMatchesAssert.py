import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_failUnlessMatchesAssert(self):
    """
        The C{failUnless*} test methods are a subset of the C{assert*} test
        methods.  This is intended to ensure that methods using the
        I{failUnless} naming scheme are not added without corresponding methods
        using the I{assert} naming scheme.  The I{assert} naming scheme is
        preferred, and new I{assert}-prefixed methods may be added without
        corresponding I{failUnless}-prefixed methods.
        """
    asserts = set(self._getAsserts())
    failUnlesses = set(prefixedMethods(self, 'failUnless'))
    self.assertEqual(failUnlesses, asserts.intersection(failUnlesses))