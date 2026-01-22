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
def test_moveToSizeCache(self, hook: Callable[[], None]=lambda: None) -> None:
    """
        L{FilePath.moveTo} clears its destination's status cache, such that
        calls to L{FilePath.getsize} after the call to C{moveTo} will report the
        new size, not the old one.

        This is a separate test from C{test_moveToExistsCache} because it is
        intended to cover the fact that the destination's cache is dropped;
        test_moveToExistsCache doesn't cover this case because (currently) a
        file that doesn't exist yet does not cache the fact of its non-
        existence.
        """
    fp = filepath.FilePath(self.mktemp())
    fp2 = filepath.FilePath(self.mktemp())
    fp.setContent(b'1234')
    fp2.setContent(b'1234567890')
    hook()
    self.assertEqual(fp.getsize(), 4)
    self.assertEqual(fp2.getsize(), 10)
    os.remove(fp2.path)
    self.assertEqual(fp2.getsize(), 10)
    fp.moveTo(fp2)
    self.assertEqual(fp2.getsize(), 4)