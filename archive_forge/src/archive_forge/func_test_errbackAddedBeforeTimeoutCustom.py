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
def test_errbackAddedBeforeTimeoutCustom(self) -> None:
    """
        An errback added before a timeout is added with a custom
        timeout function errbacks with a L{defer.CancelledError} when
        the timeout fires.  The timeout function runs if the errback
        returns the L{defer.CancelledError}.
        """
    clock = Clock()
    d: Deferred[None] = Deferred()
    dErrbacked = None

    def errback(f: Failure) -> Failure:
        nonlocal dErrbacked
        dErrbacked = f
        return f
    d.addErrback(errback)
    d.addTimeout(10, clock, _overrideFunc)
    clock.advance(15)
    assert dErrbacked is not None
    self.assertIsInstance(dErrbacked, Failure)
    self.assertIsInstance(dErrbacked.value, defer.CancelledError)
    self.assertEqual('OVERRIDDEN', self.successResultOf(d))