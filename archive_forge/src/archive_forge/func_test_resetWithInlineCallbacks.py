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
def test_resetWithInlineCallbacks(self) -> None:
    """
        When an inlineCallbacks function resumes, we should be able to reset() a
        contextvar that was set when it was first called.
        """
    clock = Clock()
    var: contextvars.ContextVar[int] = contextvars.ContextVar('testvar')

    @defer.inlineCallbacks
    def yieldingDeferred() -> Generator[Deferred[Any], Any, None]:
        token = var.set(3)
        d: Deferred[int] = Deferred()
        clock.callLater(1, d.callback, True)
        yield d
        self.assertEqual(var.get(), 3)
        var.reset(token)
        self.assertEqual(var.get(), 2)
    var.set(2)
    d = yieldingDeferred()
    clock.advance(1)
    self.successResultOf(d)