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
def test_mixedTypeChildSearchPreauth(self) -> None:
    """
        C{childSearchPreauth} called with L{bytes} on a L{unicode}-mode
        L{FilePath} will return a L{bytes}-mode L{FilePath}.
        """
    fp = filepath.FilePath('./mon€y')
    fp.createDirectory()
    self.addCleanup(lambda: fp.remove())
    child = fp.child('text.txt')
    child.touch()
    newPath = fp.childSearchPreauth(b'text.txt')
    assert newPath is not None
    self.assertIsInstance(newPath, filepath.FilePath)
    self.assertIsInstance(newPath.path, bytes)