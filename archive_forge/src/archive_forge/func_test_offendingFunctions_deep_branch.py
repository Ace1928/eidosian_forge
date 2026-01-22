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
def test_offendingFunctions_deep_branch(self) -> None:
    """
        In Python 3.6 the dis.findlinestarts documented behaviour
        was changed such that the reported lines might not be sorted ascending.
        In Python 3.10 PEP 626 introduced byte-code change such that the last
        line of a function wasn't always associated with the last byte-code.
        In the past flushWarning was not detecting that such a function was
        associated with any warnings.
        """

    def foo(a: int=1, b: int=1) -> None:
        if a:
            if b:
                warnings.warn('oh no')
            else:
                pass
    foo()
    self.assertEqual(len(self.flushWarnings([foo])), 1)