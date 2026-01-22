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
def test_cancelFiredOnOneCallbackDeferredList(self) -> None:
    """
        When a L{DeferredList} has fired because one L{Deferred} in
        the list fired with a non-failure result, the cancellation will do
        nothing instead of cancelling the rest of the L{Deferred}s.
        """
    deferredOne: Deferred[None] = Deferred()
    deferredTwo: Deferred[None] = Deferred()
    deferredList = DeferredList([deferredOne, deferredTwo], fireOnOneCallback=True)
    deferredOne.callback(None)
    deferredList.cancel()
    self.assertNoResult(deferredTwo)