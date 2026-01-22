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
def test_cleared(self) -> None:
    """
        After a particular warning event has been returned by C{flushWarnings},
        it is not returned by subsequent calls.
        """
    message = 'the message'
    category = RuntimeWarning
    warnings.warn(message=message, category=category)
    self.assertDictSubsets(self.flushWarnings(), [{'category': category, 'message': message}])
    self.assertEqual(self.flushWarnings(), [])