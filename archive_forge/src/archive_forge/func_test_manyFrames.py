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
def test_manyFrames(self) -> None:
    """
        A C{_Traceback} object constructed with multiple frames should be able
        to be passed to L{traceback.extract_tb}, and we should get a list
        containing a tuple for each frame.
        """
    tb = failure._Traceback([['caller1', 'filename.py', 7, {}, {}], ['caller2', 'filename.py', 8, {}, {}]], [['method1', 'filename.py', 123, {}, {}], ['method2', 'filename.py', 235, {}, {}]])
    self.assertEqual(traceback.extract_tb(tb), [_tb('filename.py', 123, 'method1', None), _tb('filename.py', 235, 'method2', None)])
    self.assertEqual(traceback.extract_stack(tb.tb_frame), [_tb('filename.py', 7, 'caller1', None), _tb('filename.py', 8, 'caller2', None), _tb('filename.py', 123, 'method1', None)])