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
@skipIf(symlinkSkip, 'Platform does not support symlinks')
def test_crossMountMoveToWithoutSymlink(self) -> None:
    """
        Verify that moveTo called with followLinks=False actually create
        another symlink.
        """
    invokedWith = self.setUpFaultyRename()
    f2 = self.path.child(b'file2')
    f3 = self.path.child(b'file3')
    os.symlink(self.path.child(b'file1').path, f2.path)
    f2.moveTo(f3, followLinks=False)
    self.assertTrue(f3.islink())
    self.assertEqual(f3.getContent(), b'file 1')
    self.assertTrue(invokedWith)