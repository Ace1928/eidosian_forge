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
def test_filterOnOffendingFunction(self) -> None:
    """
        The list returned by C{flushWarnings} includes only those
        warnings which refer to the source of the function passed as the value
        for C{offendingFunction}, if a value is passed for that parameter.
        """
    firstMessage = 'first warning text'
    firstCategory = UserWarning

    def one() -> None:
        warnings.warn(firstMessage, firstCategory, stacklevel=1)
    secondMessage = 'some text'
    secondCategory = RuntimeWarning

    def two() -> None:
        warnings.warn(secondMessage, secondCategory, stacklevel=1)
    one()
    two()
    self.assertDictSubsets(self.flushWarnings(offendingFunctions=[one]), [{'category': firstCategory, 'message': firstMessage}])
    self.assertDictSubsets(self.flushWarnings(offendingFunctions=[two]), [{'category': secondCategory, 'message': secondMessage}])