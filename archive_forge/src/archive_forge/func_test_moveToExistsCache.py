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
def test_moveToExistsCache(self) -> None:
    """
        A L{FilePath} that has been moved aside with L{FilePath.moveTo} no
        longer registers as existing.  Its previously non-existent target
        exists, though, as it was created by the call to C{moveTo}.
        """
    fp = filepath.FilePath(self.mktemp())
    fp2 = filepath.FilePath(self.mktemp())
    fp.touch()
    self.assertTrue(fp.exists())
    self.assertFalse(fp2.exists())
    fp.moveTo(fp2)
    self.assertFalse(fp.exists())
    self.assertTrue(fp2.exists())