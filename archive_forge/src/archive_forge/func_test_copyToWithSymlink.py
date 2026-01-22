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
def test_copyToWithSymlink(self) -> None:
    """
        Verify that copying with followLinks=True copies symlink targets
        instead of symlinks
        """
    os.symlink(self.path.child(b'sub1').path, self.path.child(b'link1').path)
    fp = filepath.FilePath(self.mktemp())
    self.path.copyTo(fp)
    self.assertFalse(fp.child(b'link1').islink())
    self.assertEqual([x.basename() for x in fp.child(b'sub1').children()], [x.basename() for x in fp.child(b'link1').children()])