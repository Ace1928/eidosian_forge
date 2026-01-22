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
def test_deprecatedAttributeHelper(self):
    """
        L{twisted.python.deprecate._DeprecatedAttribute} correctly sets its
        __name__ to match that of the deprecated attribute and emits a warning
        when the original attribute value is accessed.
        """
    name = 'ANOTHER_DEPRECATED_ATTRIBUTE'
    setattr(deprecatedattributes, name, 42)
    attr = deprecate._DeprecatedAttribute(deprecatedattributes, name, self.version, self.message)
    self.assertEqual(attr.__name__, name)

    def addStackLevel():
        attr.get()
    addStackLevel()
    warningsShown = self.flushWarnings([self.test_deprecatedAttributeHelper])
    self.assertIs(warningsShown[0]['category'], DeprecationWarning)
    self.assertEqual(warningsShown[0]['message'], self._getWarningString(name))
    self.assertEqual(len(warningsShown), 1)