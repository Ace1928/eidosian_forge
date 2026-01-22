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
def test_fromCoroutineRequiresCoroutine(self) -> None:
    """
        L{Deferred.fromCoroutine} requires a coroutine object or a generator,
        and will reject things that are not that.
        """
    thingsThatAreNotCoroutines = [lambda x: x, 1, True, self.test_fromCoroutineRequiresCoroutine, None, defer]
    for thing in thingsThatAreNotCoroutines:
        self.assertRaises(defer.NotACoroutineError, Deferred.fromCoroutine, thing)