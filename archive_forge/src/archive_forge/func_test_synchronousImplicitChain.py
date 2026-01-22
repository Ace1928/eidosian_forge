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
def test_synchronousImplicitChain(self) -> None:
    """
        If a first L{Deferred} with a result is returned from a callback on a
        second L{Deferred}, the result of the second L{Deferred} becomes the
        result of the first L{Deferred} and the result of the first L{Deferred}
        becomes L{None}.
        """
    result = object()
    first = defer.succeed(result)
    second: Deferred[None] = Deferred()
    second.addCallback(lambda ign: first)
    second.callback(None)
    results: List[Optional[object]] = []
    first.addCallback(results.append)
    self.assertIsNone(results[0])
    second.addCallback(results.append)
    self.assertIs(results[1], result)