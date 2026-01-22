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
def test_inlineCallbacksCancelCaptured(self) -> None:
    """
        Cancelling an L{defer.inlineCallbacks} correctly handles the function
        catching the L{defer.CancelledError}.

        The desired behavior is:
            1. If the function is waiting on an inner deferred, that inner
               deferred is cancelled, and a L{defer.CancelledError} is raised
               within the function.
            2. If the function catches that exception, execution continues, and
               the deferred returned by the function is not resolved.
            3. Cancelling the deferred again cancels any deferred the function
               is waiting on, and the exception is raised.
        """
    canceller1Calls: List[Deferred[object]] = []
    canceller2Calls: List[Deferred[object]] = []
    d1: Deferred[object] = Deferred(canceller1Calls.append)
    d2: Deferred[object] = Deferred(canceller2Calls.append)

    @defer.inlineCallbacks
    def testFunc() -> Generator[Deferred[object], object, None]:
        try:
            yield d1
        except Exception:
            pass
        yield d2
    funcD = testFunc()
    self.assertNoResult(d1)
    self.assertNoResult(d2)
    self.assertNoResult(funcD)
    self.assertEqual(canceller1Calls, [])
    self.assertEqual(canceller1Calls, [])
    funcD.cancel()
    self.assertEqual(canceller1Calls, [d1])
    self.assertEqual(canceller2Calls, [])
    self.assertNoResult(funcD)
    funcD.cancel()
    failure = self.failureResultOf(funcD)
    self.assertEqual(failure.type, defer.CancelledError)
    self.assertEqual(canceller2Calls, [d2])