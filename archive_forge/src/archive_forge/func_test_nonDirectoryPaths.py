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
def test_nonDirectoryPaths(self) -> None:
    """
        Verify that L{modules.walkModules} ignores entries in sys.path which
        refer to regular files in the filesystem.
        """
    existentPath = self.pathEntryWithOnePackage()
    nonDirectoryPath = FilePath(self.mktemp())
    self.assertFalse(nonDirectoryPath.exists())
    nonDirectoryPath.setContent(b'zip file or whatever\n')
    self.replaceSysPath([existentPath.path])
    beforeModules = list(modules.walkModules())
    sys.path.append(nonDirectoryPath.path)
    afterModules = list(modules.walkModules())
    self.assertEqual(beforeModules, afterModules)