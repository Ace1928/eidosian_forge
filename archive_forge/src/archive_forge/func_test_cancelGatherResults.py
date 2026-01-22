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
def test_cancelGatherResults(self) -> None:
    """
        When cancelling the L{defer.gatherResults} call, all the
        L{Deferred}s in the list will be cancelled.
        """
    deferredOne: Deferred[None] = Deferred()
    deferredTwo: Deferred[None] = Deferred()
    result = defer.gatherResults([deferredOne, deferredTwo])
    result.cancel()
    self.failureResultOf(deferredOne, defer.CancelledError)
    self.failureResultOf(deferredTwo, defer.CancelledError)
    gatherResultsFailure = self.failureResultOf(result, defer.FirstError)
    firstError = gatherResultsFailure.value
    self.assertTrue(firstError.subFailure.check(defer.CancelledError))