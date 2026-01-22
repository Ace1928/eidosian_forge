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
def test_resultFromCancel(self) -> None:
    """
        If one of the input Deferreds has a cancel function that fires it
        with success, nothing bad happens.
        """
    winner: Deferred[object] = Deferred()
    ds: list[Deferred[object]] = [winner, Deferred(canceller=lambda d: d.callback(object()))]
    expected = object()
    raceResult = race(ds)
    winner.callback(expected)
    assert_that(self.successResultOf(raceResult), equal_to((0, expected)))