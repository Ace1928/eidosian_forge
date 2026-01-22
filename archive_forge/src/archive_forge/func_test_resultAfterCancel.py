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
@given(beforeWinner=integers(min_value=0, max_value=3), afterWinner=integers(min_value=0, max_value=3))
def test_resultAfterCancel(self, beforeWinner: int, afterWinner: int) -> None:
    """
        If one of the Deferreds fires after it was cancelled its result
        goes nowhere.  In particular, it does not cause any errors to be
        logged.
        """
    ds: list[Deferred[None]] = [Deferred() for n in range(beforeWinner + 2 + afterWinner)]
    raceResult = race(ds)
    ds[beforeWinner].callback(None)
    ds[beforeWinner + 1].callback(None)
    self.successResultOf(raceResult)
    assert_that(self.flushLoggedErrors(), empty())