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
def test_cleanPendingReturnsDelayedCallStrings(self) -> None:
    """
        The Janitor produces string representations of delayed calls from the
        delayed call cleanup method. It gets the string representations
        *before* cancelling the calls; this is important because cancelling the
        call removes critical debugging information from the string
        representation.
        """
    delayedCall = DelayedCall(300, lambda: None, (), {}, lambda x: None, lambda x: None, seconds=lambda: 0)
    delayedCallString = str(delayedCall)
    reactor = StubReactor([delayedCall])
    jan = _Janitor(None, None, reactor=reactor)
    strings = jan._cleanPending()
    self.assertEqual(strings, [delayedCallString])