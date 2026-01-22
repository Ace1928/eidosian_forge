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
def test_oldTimesAreInts(self) -> None:
    """
        Verify that all times returned from the various time functions are
        integers, for compatibility.
        """
    for p in (self.path, self.path.child(b'file1')):
        self.assertEqual(type(p.getatime()), int)
        self.assertEqual(type(p.getmtime()), int)
        self.assertEqual(type(p.getctime()), int)