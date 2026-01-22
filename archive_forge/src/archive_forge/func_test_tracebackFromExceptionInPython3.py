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
def test_tracebackFromExceptionInPython3(self) -> None:
    """
        If a L{failure.Failure} is constructed with an exception but no
        traceback in Python 3, the traceback will be extracted from the
        exception's C{__traceback__} attribute.
        """
    try:
        1 / 0
    except BaseException:
        klass, exception, tb = sys.exc_info()
    f = failure.Failure(exception)
    self.assertIs(f.tb, tb)