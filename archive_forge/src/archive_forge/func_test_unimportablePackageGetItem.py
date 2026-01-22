from __future__ import annotations
import compileall
import itertools
import sys
import zipfile
from importlib.abc import PathEntryFinder
from types import ModuleType
from typing import Any, Generator
from typing_extensions import Protocol
import twisted
from twisted.python import modules
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.python.test.test_zippath import zipit
from twisted.trial.unittest import TestCase
def test_unimportablePackageGetItem(self) -> None:
    """
        If a package has been explicitly forbidden from importing by setting a
        L{None} key in sys.modules under its name,
        L{modules.PythonPath.__getitem__} should still be able to retrieve an
        unloaded L{modules.PythonModule} for that package.
        """
    shouldNotLoad: list[str] = []
    path = modules.PythonPath(sysPath=[self.pathEntryWithOnePackage().path], moduleLoader=shouldNotLoad.append, importerCache={}, sysPathHooks={}, moduleDict={'test_package': None})
    self.assertEqual(shouldNotLoad, [])
    self.assertFalse(path['test_package'].isLoaded())