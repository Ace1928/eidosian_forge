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
def test_printTracebackCapturedVars(self) -> None:
    """
        L{printTraceback} returns a traceback when called on a L{Failure}
        constructed with C{captureVars=True}.

        Local variables on the stack can not be seen in the resulting
        traceback.
        """
    self.assertDefaultTraceback(captureVars=True)