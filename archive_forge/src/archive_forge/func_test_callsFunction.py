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
def test_callsFunction(self) -> None:
    """
        L{_collectWarnings} returns the result of calling the callable passed to
        it with the parameters given.
        """
    arguments = []
    value = object()

    def f(*args: object, **kwargs: object) -> object:
        arguments.append((args, kwargs))
        return value
    result = _collectWarnings(lambda x: None, f, 1, 'a', b=2, c='d')
    self.assertEqual(arguments, [((1, 'a'), {'b': 2, 'c': 'd'})])
    self.assertIdentical(result, value)