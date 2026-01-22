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
def test_changed(self) -> None:
    """
        L{FilePath.changed} indicates that the L{FilePath} has changed, but does
        not re-read the status information from the filesystem until it is
        queried again via another method, such as C{getsize}.
        """
    fp = filepath.FilePath(self.mktemp())
    fp.setContent(b'12345')
    self.assertEqual(fp.getsize(), 5)
    with open(fp.path, 'wb') as fObj:
        fObj.write(b'12345678')
    self.assertEqual(fp.getsize(), 5)
    fp.changed()
    self.assertEqual(fp.getsize(), 8)