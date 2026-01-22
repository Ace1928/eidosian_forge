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
def test_test(self) -> None:
    """
        Self-test for assertNotUnequal to make sure the assertion works.
        """
    with self.assertRaises(AssertionError) as ae:
        self.assertNotUnequal(3, 4, 'custom message')
    self.assertIn('__ne__ not implemented correctly', str(ae.exception))
    self.assertIn('custom message', str(ae.exception))
    with self.assertRaises(AssertionError) as ae2:
        self.assertNotUnequal(4, 3)
    self.assertIn('__ne__ not implemented correctly', str(ae2.exception))
    self.assertNotUnequal(3, 3)