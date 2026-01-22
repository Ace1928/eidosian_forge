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
@defer.inlineCallbacks
def testFunction() -> Generator[Deferred[Any], Any, None]:
    self.assertEqual(var.get(), 2)
    yield defer.succeed(1)
    self.assertEqual(var.get(), 2)
    clock.callLater(1, mutatingDeferred.callback, True)
    yield mutatingDeferred
    self.assertEqual(var.get(), 2)
    clock.callLater(1, mutatingDeferredThatFails.callback, True)
    try:
        yield mutatingDeferredThatFails
    except Exception:
        self.assertEqual(var.get(), 2)
    else:
        raise Exception('???? should have failed')
    yield yieldingDeferred()
    self.assertEqual(var.get(), 2)
    defer.returnValue(True)