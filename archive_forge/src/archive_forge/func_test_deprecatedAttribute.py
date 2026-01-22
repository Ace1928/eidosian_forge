import inspect
import sys
import types
import warnings
from os.path import normcase
from warnings import catch_warnings, simplefilter
from incremental import Version
from twisted.python import deprecate
from twisted.python.deprecate import (
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.python.test import deprecatedattributes
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SynchronousTestCase
from twisted.python.deprecate import deprecatedModuleAttribute
from incremental import Version
from twisted.python import deprecate
from twisted.python import deprecate
def test_deprecatedAttribute(self):
    """
        L{twisted.python.deprecate.deprecatedModuleAttribute} wraps a
        module-level attribute in an object that emits a deprecation warning
        when it is accessed the first time only, while leaving other unrelated
        attributes alone.
        """
    deprecatedattributes.ANOTHER_ATTRIBUTE
    warningsShown = self.flushWarnings([self.test_deprecatedAttribute])
    self.assertEqual(len(warningsShown), 0)
    name = 'DEPRECATED_ATTRIBUTE'
    getattr(deprecatedattributes, name)
    warningsShown = self.flushWarnings([self.test_deprecatedAttribute])
    self.assertEqual(len(warningsShown), 1)
    self.assertIs(warningsShown[0]['category'], DeprecationWarning)
    self.assertEqual(warningsShown[0]['message'], self._getWarningString(name))