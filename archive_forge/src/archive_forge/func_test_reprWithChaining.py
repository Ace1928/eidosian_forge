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
def test_reprWithChaining(self) -> None:
    """
        If a L{Deferred} C{a} has been fired, but is waiting on another
        L{Deferred} C{b} that appears in its callback chain, then C{repr(a)}
        says that it is waiting on C{b}.
        """
    a: Deferred[None] = Deferred()
    b: Deferred[None] = Deferred()
    b.chainDeferred(a)
    self.assertEqual(repr(a), f'<Deferred at 0x{id(a):x} waiting on Deferred at 0x{id(b):x}>')