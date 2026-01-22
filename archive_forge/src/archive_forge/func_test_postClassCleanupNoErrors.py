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
def test_postClassCleanupNoErrors(self) -> None:
    """
        The post-class cleanup method will not call C{addError} on the result
        if there are no pending calls or selectables.
        """
    reactor = StubReactor([])
    test = object()
    reporter = StubErrorReporter()
    jan = _Janitor(test, reporter, reactor=reactor)
    jan.postClassCleanup()
    self.assertEqual(reporter.errors, [])