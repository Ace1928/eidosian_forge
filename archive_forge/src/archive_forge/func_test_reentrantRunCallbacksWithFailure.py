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
def test_reentrantRunCallbacksWithFailure(self) -> None:
    """
        After an exception is raised by a callback which was added to a
        L{Deferred} by a callback on that L{Deferred}, the L{Deferred} should
        call the first errback with a L{Failure} wrapping that exception.
        """
    exceptionMessage = 'callback raised exception'
    deferred: Deferred[None] = Deferred()

    def callback2(result: object) -> None:
        raise Exception(exceptionMessage)

    def callback1(result: object) -> None:
        deferred.addCallback(callback2)
    deferred.addCallback(callback1)
    deferred.callback(None)
    exception = self.assertImmediateFailure(deferred, Exception)
    self.assertEqual(exception.args, (exceptionMessage,))