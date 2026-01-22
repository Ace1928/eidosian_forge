from __future__ import annotations
import sys
import warnings
from io import StringIO
from typing import Mapping, Sequence, TypeVar
from unittest import TestResult
from twisted.python.filepath import FilePath
from twisted.trial._synctest import (
from twisted.trial.unittest import SynchronousTestCase
import warnings
import warnings
def test_suppresses(self) -> None:
    """
        Any warnings emitted by a call to a function passed to
        L{_collectWarnings} are not actually emitted to the warning system.
        """
    output = StringIO()
    self.patch(sys, 'stdout', output)
    _collectWarnings(lambda x: None, warnings.warn, 'text')
    self.assertEqual(output.getvalue(), '')