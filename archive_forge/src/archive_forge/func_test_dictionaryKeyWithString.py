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
def test_dictionaryKeyWithString(self) -> None:
    """
        Verify that path instances are usable as dictionary keys which do not clash
        with their string counterparts.
        """
    f1 = self.path.child(b'file1')
    dictoid: Dict[Union[filepath.FilePath[bytes], bytes], str] = {f1: 'hello'}
    dictoid[f1.path] = 'goodbye'
    self.assertEqual(len(dictoid), 2)