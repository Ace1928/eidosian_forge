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
def test_resultOfDeferredResultOfDeferredOfFiredDeferredCalled(self) -> None:
    """
        Given three Deferreds, one chained to the next chained to the next,
        callbacks on the middle Deferred which are added after the chain is
        created are called once the last Deferred fires.

        This is more of a regression-style test.  It doesn't exercise any
        particular code path through the current implementation of Deferred, but
        it does exercise a broken codepath through one of the variations of the
        implementation proposed as a resolution to ticket #411.
        """
    first: Deferred[None] = Deferred()
    second: Deferred[None] = Deferred()
    third: Deferred[None] = Deferred()
    first.addCallback(lambda ignored: second)
    second.addCallback(lambda ignored: third)
    second.callback(None)
    first.callback(None)
    third.callback(None)
    results: List[None] = []
    second.addCallback(results.append)
    self.assertEqual(results, [None])