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
def test_reentrantRunCallbacks(self) -> None:
    """
        A callback added to a L{Deferred} by a callback on that L{Deferred}
        should be added to the end of the callback chain.
        """
    deferred: Deferred[None] = Deferred()
    called = []

    def callback3(result: None) -> None:
        called.append(3)

    def callback2(result: None) -> None:
        called.append(2)

    def callback1(result: None) -> None:
        called.append(1)
        deferred.addCallback(callback3)
    deferred.addCallback(callback1)
    deferred.addCallback(callback2)
    deferred.callback(None)
    self.assertEqual(called, [1, 2, 3])