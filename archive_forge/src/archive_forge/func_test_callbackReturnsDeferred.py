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
def test_callbackReturnsDeferred(self) -> None:
    """
        Callbacks passed to L{Deferred.addCallback} can return Deferreds and
        the next callback will not be run until that Deferred fires.
        """
    d1: Deferred[int] = Deferred()
    d2: Deferred[int] = Deferred()
    d2.pause()
    d1.addCallback(lambda r: d2)
    d1.addCallback(self._callback)
    d1.callback(1)
    self.assertIsNone(self.callbackResults, 'Should not have been called yet.')
    d2.callback(2)
    self.assertIsNone(self.callbackResults, 'Still should not have been called yet.')
    d2.unpause()
    self.assertIsNotNone(self.callbackResults, 'Should have been called now')
    assert self.callbackResults is not None, 'make that legible to the type checker'
    self.assertEquals(self.callbackResults[0][0], 2, 'Result should have been from second deferred:{}'.format(self.callbackResults))