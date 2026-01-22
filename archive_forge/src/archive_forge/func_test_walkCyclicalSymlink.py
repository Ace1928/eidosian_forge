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
def test_walkCyclicalSymlink(self) -> None:
    """
        Verify that walking a path with a cyclical symlink raises an error
        """
    self.createLinks()
    os.symlink(self.path.child(b'sub1').path, self.path.child(b'sub1').child(b'sub1.loopylink').path)

    def iterateOverPath() -> List[bytes]:
        return [foo.path for foo in self.path.walk()]
    self.assertRaises(filepath.LinkError, iterateOverPath)