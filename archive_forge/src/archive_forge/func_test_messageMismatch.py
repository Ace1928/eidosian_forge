import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_messageMismatch(self):
    """
        L{getDeprecatedModuleAttribute} fails the test if the I{message} isn't
        part of the deprecation message prefix.
        """
    self.assertRaises(self.failureException, self.getDeprecatedModuleAttribute, __name__, 'somethingOld', self.version, "It's shiny and new")
    self.assertEqual([], self.flushWarnings())