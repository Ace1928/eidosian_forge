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
def test_rwxShorthand(self) -> None:
    """
        L{RWX}'s shorthand string should be 'rwx' if read, write, and execute
        permission bits are true.  If any of those permissions bits are false,
        the character is replaced by a '-'.
        """

    def getChar(val: bool, letter: str) -> str:
        if val:
            return letter
        return '-'
    for r in (True, False):
        for w in (True, False):
            for x in (True, False):
                rwx = filepath.RWX(r, w, x)
                self.assertEqual(rwx.shorthand(), getChar(r, 'r') + getChar(w, 'w') + getChar(x, 'x'))
    self.assertEqual(filepath.RWX(True, False, True).shorthand(), 'r-x')