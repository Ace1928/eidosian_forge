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
def test_failureBeforeTimeout(self) -> None:
    """
        The L{Deferred} errbacks with the failure if it fails before the
        timeout. No cancellation happens after the errback either, which
        could also cancel inner deferreds.
        """
    clock = Clock()
    d: Deferred[None] = Deferred()
    d.addTimeout(10, clock)
    innerDeferred: Deferred[None] = Deferred()
    dErrbacked: Optional[Failure] = None
    error = ValueError('fail')

    def onErrback(f: Failure) -> Deferred[None]:
        nonlocal dErrbacked
        dErrbacked = f
        return innerDeferred
    d.addErrback(onErrback)
    d.errback(error)
    assert dErrbacked is not None
    self.assertIsInstance(dErrbacked, Failure)
    self.assertIs(dErrbacked.value, error)
    clock.advance(15)
    self.assertNoResult(innerDeferred)