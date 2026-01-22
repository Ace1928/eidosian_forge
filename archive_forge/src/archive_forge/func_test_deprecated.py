import unittest as pyunit
import warnings
from incremental import Version, getVersionString
from twisted.internet.defer import Deferred, fail, succeed
from twisted.python.deprecate import deprecated, deprecatedModuleAttribute
from twisted.python.failure import Failure
from twisted.python.reflect import (
from twisted.python.util import FancyEqMixin
from twisted.trial import unittest
def test_deprecated(self):
    """
        L{getDeprecatedModuleAttribute} returns the specified attribute and
        consumes the deprecation warning that generates.
        """
    self.assertIs(_somethingOld, self.getDeprecatedModuleAttribute(__name__, 'somethingOld', self.version))
    self.assertEqual([], self.flushWarnings())