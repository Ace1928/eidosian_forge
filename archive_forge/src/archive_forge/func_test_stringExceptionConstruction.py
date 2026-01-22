from __future__ import annotations
import linecache
import pdb
import re
import sys
import traceback
from dis import distb
from io import StringIO
from traceback import FrameSummary
from types import TracebackType
from typing import Any, Generator
from unittest import skipIf
from cython_test_exception_raiser import raiser
from twisted.python import failure, reflect
from twisted.trial.unittest import SynchronousTestCase
def test_stringExceptionConstruction(self) -> None:
    """
        Constructing a C{Failure} with a string as its exception value raises
        a C{TypeError}, as this is no longer supported as of Python 2.6.
        """
    exc = self.assertRaises(TypeError, failure.Failure, 'ono!')
    self.assertIn('Strings are not supported by Failure', str(exc))