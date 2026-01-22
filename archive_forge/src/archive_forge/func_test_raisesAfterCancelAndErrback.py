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
def test_raisesAfterCancelAndErrback(self) -> None:
    """
        A L{Deferred} without a canceller, when cancelled must allow
        a single extra call to errback, and raise
        L{defer.AlreadyCalledError} if callbacked or errbacked thereafter.
        """
    d: Deferred[None] = Deferred()
    d.addCallbacks(self._callback, self._errback)
    d.cancel()
    d.errback(Exception())
    self.assertRaises(defer.AlreadyCalledError, d.callback, None)
    self.assertRaises(defer.AlreadyCalledError, d.errback, Exception())