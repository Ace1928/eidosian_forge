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
def test_multipleUsages(self) -> Deferred[None]:
    """
        Test that a DeferredFilesystemLock can be used multiple times
        """

    def lockAquired(ign: object) -> Deferred[None]:
        self.lock.unlock()
        d = self.lock.deferUntilLocked()
        return d
    self.lock.lock()
    self.clock.callLater(1, self.lock.unlock)
    d = self.lock.deferUntilLocked()
    d.addCallback(lockAquired)
    self.clock.advance(1)
    return d