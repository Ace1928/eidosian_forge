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
def test_rwxFromBools(self) -> None:
    """
        L{RWX}'s constructor takes a set of booleans
        """
    for r in (True, False):
        for w in (True, False):
            for x in (True, False):
                rwx = filepath.RWX(r, w, x)
                self.assertEqual(rwx.read, r)
                self.assertEqual(rwx.write, w)
                self.assertEqual(rwx.execute, x)
    rwx = filepath.RWX(True, True, True)
    self.assertTrue(rwx.read and rwx.write and rwx.execute)