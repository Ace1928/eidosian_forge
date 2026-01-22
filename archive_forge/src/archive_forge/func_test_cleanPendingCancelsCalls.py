from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_cleanPendingCancelsCalls(self) -> None:
    """
        During pending-call cleanup, the janitor cancels pending timed calls.
        """

    def func() -> str:
        return 'Lulz'
    cancelled: list[DelayedCall] = []
    delayedCall = DelayedCall(300, func, (), {}, cancelled.append, lambda x: None)
    reactor = StubReactor([delayedCall])
    jan = _Janitor(None, None, reactor=reactor)
    jan._cleanPending()
    self.assertEqual(cancelled, [delayedCall])