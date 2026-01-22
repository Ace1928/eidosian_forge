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
def test_twistedShowsUp(self) -> None:
    """
        Scrounge around in the top-level module namespace and make sure that
        Twisted shows up, and that the module thusly obtained is the same as
        the module that we find when we look for it explicitly by name.
        """
    self.assertEqual(modules.getModule('twisted'), self.findByIteration('twisted'))