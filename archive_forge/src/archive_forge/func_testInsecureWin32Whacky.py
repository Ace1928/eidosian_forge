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
@skipIf(platform.getType() != 'win32', 'Test will run only on Windows.')
def testInsecureWin32Whacky(self) -> None:
    """
        Windows has 'special' filenames like NUL and CON and COM1 and LPR
        and PRN and ... god knows what else.  They can be located anywhere in
        the filesystem.  For obvious reasons, we do not wish to normally permit
        access to these.
        """
    self.assertRaises(filepath.InsecurePath, self.path.child, b'CON')
    self.assertRaises(filepath.InsecurePath, self.path.child, b'C:CON')
    self.assertRaises(filepath.InsecurePath, self.path.child, 'C:\\CON')