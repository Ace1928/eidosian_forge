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
def test_errorLogNoRepr(self) -> None:
    """
        Verify that when a L{Deferred} with no references to it is fired,
        the logged message does not contain a repr of the failure object.
        """
    Deferred().addCallback(lambda x: 1 // 0).callback(1)
    gc.collect()
    self._check()
    self.assertEqual(2, len(self.c))
    msg = log.textFromEventDict(self.c[-1])
    assert msg is not None
    expected = 'Unhandled Error\nTraceback '
    self.assertTrue(msg.startswith(expected), f'Expected message starting with: {expected!r}')