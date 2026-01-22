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
def test_nonexistentPaths(self) -> None:
    """
        Verify that L{modules.walkModules} ignores entries in sys.path which
        do not exist in the filesystem.
        """
    existentPath = self.pathEntryWithOnePackage()
    nonexistentPath = FilePath(self.mktemp())
    self.assertFalse(nonexistentPath.exists())
    self.replaceSysPath([existentPath.path])
    expected = [modules.getModule('test_package')]
    beforeModules = list(modules.walkModules())
    sys.path.append(nonexistentPath.path)
    afterModules = list(modules.walkModules())
    self.assertEqual(beforeModules, expected)
    self.assertEqual(afterModules, expected)