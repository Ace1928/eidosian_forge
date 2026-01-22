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
def test_getDeprecationWarningStringReplacementWithCallable(self):
    """
        L{getDeprecationWarningString} takes an additional replacement parameter
        that can be used to add information to the deprecation. If the
        replacement parameter is a callable, its fully qualified name will be
        interpolated into the result.
        """
    version = Version('Twisted', 8, 0, 0)
    warningString = getDeprecationWarningString(self.test_getDeprecationWarningString, version, replacement=dummyReplacementMethod)
    self.assertEqual(warningString, '%s was deprecated in Twisted 8.0.0; please use %s.dummyReplacementMethod instead' % (fullyQualifiedName(self.test_getDeprecationWarningString), __name__))