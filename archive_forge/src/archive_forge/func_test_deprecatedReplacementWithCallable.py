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
def test_deprecatedReplacementWithCallable(self):
    """
        L{deprecated} takes an additional replacement parameter that can be used
        to indicate the new, non-deprecated method developers should use.  If
        the replacement parameter is a callable, its fully qualified name will
        be interpolated into the warning message.
        """
    version = Version('Twisted', 8, 0, 0)
    decorator = deprecated(version, replacement=dummyReplacementMethod)
    dummy = decorator(dummyCallable)
    self.assertEqual(dummy.__doc__, '\n    Do nothing.\n\n    This is used to test the deprecation decorators.\n\n    Deprecated in Twisted 8.0.0; please use %s.dummyReplacementMethod instead.\n    ' % (__name__,))