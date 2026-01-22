from __future__ import annotations
import contextvars
import functools
import gc
import re
import traceback
import types
import unittest as pyunit
import warnings
import weakref
from asyncio import (
from typing import (
from hamcrest import assert_that, empty, equal_to
from hypothesis import given
from hypothesis.strategies import integers
from typing_extensions import assert_type
from twisted.internet import defer, reactor
from twisted.internet.defer import (
from twisted.internet.task import Clock
from twisted.python import log
from twisted.python.compat import _PYPY
from twisted.python.failure import Failure
from twisted.trial import unittest
def test_inlineCallbacksTracebacks(self) -> None:
    """
        L{defer.inlineCallbacks} that re-raise tracebacks into their deferred
        should not lose their tracebacks.
        """
    f = getDivisionFailure()
    d: Deferred[None] = Deferred()
    try:
        f.raiseException()
    except BaseException:
        d.errback()

    def ic(d: object) -> Generator[Any, Any, None]:
        yield d
    defer.inlineCallbacks(ic)
    newFailure = self.failureResultOf(d)
    tb = traceback.extract_tb(newFailure.getTracebackObject())
    self.assertEqual(len(tb), 3)
    self.assertIn('test_defer', tb[2][0])
    self.assertEqual('getDivisionFailure', tb[2][2])
    self.assertEqual('1 / 0', tb[2][3])
    self.assertIn('test_defer', tb[0][0])
    self.assertEqual('test_inlineCallbacksTracebacks', tb[0][2])
    self.assertEqual('f.raiseException()', tb[0][3])