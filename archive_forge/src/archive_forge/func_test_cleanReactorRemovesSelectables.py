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
def test_cleanReactorRemovesSelectables(self) -> None:
    """
        The Janitor will remove selectables during reactor cleanup.
        """
    reactor = StubReactor([])
    jan = _Janitor(None, None, reactor=reactor)
    jan._cleanReactor()
    self.assertEqual(reactor.removeAllCalled, 1)