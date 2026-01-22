from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
def test_unicodeSiblingExtensionSearch(self) -> None:
    """
        C{siblingExtensionSearch} called with L{unicode} on a L{unicode}-mode
        L{FilePath} will return a L{list} of L{unicode}-mode L{FilePath}s.
        """
    fp = filepath.FilePath('./mon€y')
    sibling = filepath.FilePath(fp._asTextPath() + '.txt')
    sibling.touch()
    newPath = fp.siblingExtensionSearch('.txt')
    assert newPath is not None
    self.assertIsInstance(newPath, filepath.FilePath)
    self.assertIsInstance(newPath.path, str)