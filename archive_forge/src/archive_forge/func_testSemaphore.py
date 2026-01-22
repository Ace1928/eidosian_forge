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
def testSemaphore(self) -> None:
    N = 13
    sem = DeferredSemaphore(N)
    controlDeferred: Deferred[None] = Deferred()
    helperArg: object = None

    def helper(arg: object) -> Deferred[None]:
        nonlocal helperArg
        helperArg = arg
        return controlDeferred
    results: List[object] = []
    uniqueObject = object()
    resultDeferred = sem.run(helper, arg=uniqueObject)
    resultDeferred.addCallback(results.append)
    resultDeferred.addCallback(self._incr)
    self.assertEqual(results, [])
    self.assertEqual(helperArg, uniqueObject)
    controlDeferred.callback(None)
    self.assertIsNone(results.pop())
    self.assertEqual(self.counter, 1)
    self.counter = 0
    for i in range(1, 1 + N):
        sem.acquire().addCallback(self._incr)
        self.assertEqual(self.counter, i)
    success = []

    def fail(r: object) -> None:
        success.append(False)

    def succeed(r: object) -> None:
        success.append(True)
    d = sem.acquire().addCallbacks(fail, succeed)
    d.cancel()
    self.assertEqual(success, [True])
    sem.acquire().addCallback(self._incr)
    self.assertEqual(self.counter, N)
    sem.release()
    self.assertEqual(self.counter, N + 1)
    for i in range(1, 1 + N):
        sem.release()
        self.assertEqual(self.counter, N + 1)