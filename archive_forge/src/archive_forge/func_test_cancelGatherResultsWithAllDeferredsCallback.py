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
def test_cancelGatherResultsWithAllDeferredsCallback(self) -> None:
    """
        When cancelling the L{defer.gatherResults} call, if all the
        L{Deferred}s callback in their canceller, the L{Deferred}
        returned by L{defer.gatherResults} will be callbacked with the C{list}
        of the results.
        """
    deferredOne: Deferred[str] = Deferred(fakeCallbackCanceller)
    deferredTwo: Deferred[str] = Deferred(fakeCallbackCanceller)
    result = defer.gatherResults([deferredOne, deferredTwo])
    result.cancel()
    callbackResult = self.successResultOf(result)
    self.assertEqual(callbackResult[0], 'Callback Result')
    self.assertEqual(callbackResult[1], 'Callback Result')