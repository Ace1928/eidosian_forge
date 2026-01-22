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
def test_wrappedModule(self):
    """
        Deprecating an attribute in a module replaces and wraps that module
        instance, in C{sys.modules}, with a
        L{twisted.python.deprecate._ModuleProxy} instance but only if it hasn't
        already been wrapped.
        """
    sys.modules[self._testModuleName] = mod = types.ModuleType('foo')
    self.addCleanup(sys.modules.pop, self._testModuleName)
    setattr(mod, 'first', 1)
    setattr(mod, 'second', 2)
    deprecate.deprecatedModuleAttribute(Version('Twisted', 8, 0, 0), 'message', self._testModuleName, 'first')
    proxy = sys.modules[self._testModuleName]
    self.assertNotEqual(proxy, mod)
    deprecate.deprecatedModuleAttribute(Version('Twisted', 8, 0, 0), 'message', self._testModuleName, 'second')
    self.assertIs(proxy, sys.modules[self._testModuleName])