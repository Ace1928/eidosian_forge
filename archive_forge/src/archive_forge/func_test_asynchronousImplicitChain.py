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
def test_asynchronousImplicitChain(self) -> None:
    """
        If a first L{Deferred} without a result is returned from a callback on
        a second L{Deferred}, the result of the second L{Deferred} becomes the
        result of the first L{Deferred} as soon as the first L{Deferred} has
        one and the result of the first L{Deferred} becomes L{None}.
        """
    first: Deferred[object] = Deferred()
    second: Deferred[object] = Deferred()
    second.addCallback(lambda ign: first)
    second.callback(None)
    firstResult: List[object] = []
    first.addCallback(firstResult.append)
    secondResult: List[object] = []
    second.addCallback(secondResult.append)
    self.assertEqual(firstResult, [])
    self.assertEqual(secondResult, [])
    result = object()
    first.callback(result)
    self.assertEqual(firstResult, [None])
    self.assertEqual(secondResult, [result])