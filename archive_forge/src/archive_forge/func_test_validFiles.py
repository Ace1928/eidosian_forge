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
def test_validFiles(self) -> None:
    """
        Make sure that we can read existent non-empty files.
        """
    f1 = self.path.child(b'file1')
    with f1.open() as f:
        self.assertEqual(f.read(), self.f1content)
    f2 = self.path.child(b'sub1').child(b'file2')
    with f2.open() as f:
        self.assertEqual(f.read(), self.f2content)