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
def test_callbackMaybeReturnsFailure(self) -> None:
    """
        Callbacks passed to addCallback may return Failures.
        """
    d: Deferred[int] = Deferred()
    shouldFail = False
    rte = RuntimeError()

    def maybeFail(result: int) -> Union[int, Failure]:
        if shouldFail:
            return Failure(rte)
        else:
            return result + 1
    d.callback(6)
    self.assertEqual(self.successResultOf(d.addCallback(maybeFail)), 7)
    d = Deferred[int]()
    shouldFail = True
    d.callback(6)
    self.assertIs(self.failureResultOf(d.addCallback(maybeFail).addCallback(maybeFail), RuntimeError).value, rte)