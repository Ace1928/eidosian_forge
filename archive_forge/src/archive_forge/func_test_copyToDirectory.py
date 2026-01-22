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
def test_copyToDirectory(self) -> None:
    """
        L{FilePath.copyTo} makes a copy of all the contents of the directory
        named by that L{FilePath} if it is able to do so.
        """
    oldPaths = list(self.path.walk())
    fp = filepath.FilePath(self.mktemp())
    self.path.copyTo(fp)
    self.path.remove()
    fp.copyTo(self.path)
    newPaths = list(self.path.walk())
    newPaths.sort()
    oldPaths.sort()
    self.assertEqual(newPaths, oldPaths)