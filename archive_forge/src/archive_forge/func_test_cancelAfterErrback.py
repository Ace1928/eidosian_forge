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
def test_cancelAfterErrback(self) -> None:
    """
        Test that cancelling a L{Deferred} after it has been errbacked does
        not result in a L{defer.CancelledError}.
        """

    def cancel(d: Deferred[object]) -> None:
        self.cancellerCallCount += 1
        d.errback(GenericError())
    d: Deferred[None] = Deferred(canceller=cancel)
    d.addCallbacks(self._callback, self._errback)
    d.errback(GenericError())
    d.cancel()
    self.assertEqual(self.cancellerCallCount, 0)
    assert self.errbackResults is not None
    self.assertEqual(self.errbackResults.type, GenericError)
    self.assertIsNone(self.callbackResults)