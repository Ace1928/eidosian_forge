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
@skipIf(not platform.isWindows(), 'Test only works on Windows')
def test_unicodereprWindows(self) -> None:
    """
        The repr of a L{unicode} L{FilePath} shouldn't burst into flames.
        """
    fp = filepath.FilePath('C:\\')
    reprOutput = repr(fp)
    self.assertEqual("FilePath('C:\\\\')", reprOutput)