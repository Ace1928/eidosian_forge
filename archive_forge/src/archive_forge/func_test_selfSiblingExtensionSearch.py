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
def test_selfSiblingExtensionSearch(self) -> None:
    """
        C{siblingExtension} passed an empty string should return the same path,
        in the type of its argument.
        """
    exists = filepath.FilePath(self.mktemp())
    exists.touch()
    notExists = filepath.FilePath(self.mktemp())
    self.assertEqual(exists.siblingExtensionSearch(b''), exists.asBytesMode())
    self.assertEqual(exists.siblingExtensionSearch(''), exists.asTextMode())
    self.assertEqual(notExists.siblingExtensionSearch(''), None)
    self.assertEqual(notExists.siblingExtensionSearch(b''), None)