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
def test_cancelLockBeforeAcquired(self) -> None:
    """
        When canceling a L{Deferred} from a L{DeferredLock} that does not
        yet have the lock (i.e., the L{Deferred} has not fired), the cancel
        should cause a L{defer.CancelledError} failure.
        """
    lock = DeferredLock()
    lock.acquire()
    d = lock.acquire()
    d.cancel()
    self.assertImmediateFailure(d, defer.CancelledError)