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
def test_flushedWarningsConfiguredAsErrors(self) -> None:
    """
        If a warnings filter has been installed which turns warnings into
        exceptions, tests which emit those warnings but flush them do not have
        an error added to the reporter.
        """

    class CustomWarning(Warning):
        pass
    result = TestResult()
    case = Mask.MockTests('test_flushed')
    case.category = CustomWarning
    originalWarnings = warnings.filters[:]
    try:
        warnings.simplefilter('error')
        case.run(result)
        self.assertEqual(result.errors, [])
    finally:
        warnings.filters[:] = originalWarnings