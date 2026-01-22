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
def test_cancelDeferredListWithException(self) -> None:
    """
        Cancelling a L{DeferredList} will cancel every L{Deferred}
        in the list even exceptions raised from the C{cancel} method of the
        L{Deferred}s.
        """

    def cancellerRaisesException(deferred: Deferred[object]) -> None:
        """
            A L{Deferred} canceller that raises an exception.

            @param deferred: The cancelled L{Deferred}.
            """
        raise RuntimeError('test')
    deferredOne: Deferred[None] = Deferred(cancellerRaisesException)
    deferredTwo: Deferred[None] = Deferred()
    deferredList = DeferredList([deferredOne, deferredTwo])
    deferredList.cancel()
    self.failureResultOf(deferredTwo, defer.CancelledError)
    errors = self.flushLoggedErrors(RuntimeError)
    self.assertEqual(len(errors), 1)