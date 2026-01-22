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
def chainDeferreds(howMany: int) -> int:
    stack = []

    def recordStackDepth(ignored: object) -> None:
        stack.append(len(traceback.extract_stack()))
    top: Deferred[None] = Deferred()
    innerDeferreds: List[Deferred[None]] = [Deferred() for ignored in range(howMany)]
    originalInners = innerDeferreds[:]
    last: Deferred[None] = Deferred()
    inner = innerDeferreds.pop()

    def cbInner(ignored: object, inner: Deferred[None]=inner) -> Deferred[None]:
        return inner
    top.addCallback(cbInner)
    top.addCallback(recordStackDepth)
    while innerDeferreds:
        newInner = innerDeferreds.pop()

        def cbNewInner(ignored: object, inner: Deferred[None]=newInner) -> Deferred[None]:
            return inner
        inner.addCallback(cbNewInner)
        inner = newInner
    inner.addCallback(lambda ign: last)
    top.callback(None)
    for inner in originalInners:
        inner.callback(None)
    self.assertEqual(stack, [])
    last.callback(None)
    return stack[0]