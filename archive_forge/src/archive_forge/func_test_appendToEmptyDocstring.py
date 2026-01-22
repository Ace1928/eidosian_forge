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
def test_appendToEmptyDocstring(self):
    """
        Appending to an empty docstring simply replaces the docstring.
        """

    def noDocstring():
        pass
    _appendToDocstring(noDocstring, 'Appended text.')
    self.assertEqual('Appended text.', noDocstring.__doc__)