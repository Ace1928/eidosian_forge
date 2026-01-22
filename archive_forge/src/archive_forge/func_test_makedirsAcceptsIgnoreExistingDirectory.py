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
def test_makedirsAcceptsIgnoreExistingDirectory(self) -> None:
    """
        C{FilePath.makedirs} succeeds when called on a directory that already
        exists and the c{ignoreExistingDirectory} argument is set to C{True}.
        """
    fp = filepath.FilePath(self.mktemp())
    fp.makedirs()
    self.assertTrue(fp.exists())
    fp.makedirs(ignoreExistingDirectory=True)
    self.assertTrue(fp.exists())