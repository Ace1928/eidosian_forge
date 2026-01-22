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
def test_deprecatedModuleMultipleTimes(self):
    """
        If L{deprecatedModuleAttribute} is used to deprecate a module attribute
        of a package, only one deprecation warning is emitted when the
        deprecated module is subsequently imported.
        """
    mp = self.simpleModuleEntry()
    self.checkOneWarning(mp)
    self.checkOneWarning(mp)
    for x in range(2):
        self.checkOneWarning(mp)