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
def test_fakeFrameAttributes(self) -> None:
    """
        L{_Frame} instances have the C{f_globals} and C{f_locals} attributes
        bound to C{dict} instance.  They also have the C{f_code} attribute
        bound to something like a code object.
        """
    back_frame = failure._Frame(('dummyparent', 'dummyparentfile', 111, None, None), None)
    fake_locals = {'local_var': 42}
    fake_globals = {'global_var': 100}
    frame = failure._Frame(('dummyname', 'dummyfilename', 42, fake_locals, fake_globals), back_frame)
    self.assertEqual(frame.f_globals, fake_globals)
    self.assertEqual(frame.f_locals, fake_locals)
    self.assertIsInstance(frame.f_code, failure._Code)
    self.assertEqual(frame.f_back, back_frame)
    self.assertIsInstance(frame.f_builtins, dict)
    self.assertIsInstance(frame.f_lasti, int)
    self.assertEqual(frame.f_lineno, 42)
    self.assertIsInstance(frame.f_trace, type(None))