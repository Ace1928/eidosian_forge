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
def test_warningsConfiguredAsErrors(self) -> None:
    """
        If a warnings filter has been installed which turns warnings into
        exceptions, tests have an error added to the reporter for them for each
        unflushed warning.
        """

    class CustomWarning(Warning):
        pass
    result = TestResult()
    case = Mask.MockTests('test_unflushed')
    case.category = CustomWarning
    originalWarnings = warnings.filters[:]
    try:
        warnings.simplefilter('error')
        case.run(result)
        self.assertEqual(len(result.errors), 1)
        self.assertIdentical(result.errors[0][0], case)
        self.assertTrue(result.errors[0][1].splitlines()[-1].endswith('CustomWarning: some warning text'))
    finally:
        warnings.filters[:] = originalWarnings