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
def test_noCancellerMultipleCancel(self) -> None:
    """
        Calling cancel multiple times on a deferred with no canceller
        results in a L{defer.CancelledError}. Subsequent calls to cancel
        do not cause an error.
        """
    d: Deferred[None] = Deferred()
    d.addCallbacks(self._callback, self._errback)
    d.cancel()
    assert self.errbackResults is not None
    self.assertEqual(self.errbackResults.type, defer.CancelledError)
    currentFailure = self.errbackResults
    d.cancel()
    self.assertIs(currentFailure, self.errbackResults)