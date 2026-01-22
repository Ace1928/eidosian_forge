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
def test_asFutureCancelFuture(self) -> None:
    """
        L{Deferred.asFuture} returns a L{asyncio.Future} which, when
        cancelled, will cancel the original L{Deferred}.
        """
    called = False

    def canceler(dprime: Deferred[object]) -> None:
        nonlocal called
        called = True
    d: Deferred[None] = Deferred(canceler)
    loop = self.newLoop()
    aFuture = d.asFuture(loop)
    aFuture.cancel()
    callAllSoonCalls(loop)
    self.assertTrue(called)
    self.assertEqual(self.successResultOf(d), None)
    self.assertRaises(CancelledError, aFuture.result)