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
def test_excInfo(self) -> None:
    """
        L{excInfoOrFailureToExcInfo} returns exactly what it is passed, if it is
        passed a tuple like the one returned by L{sys.exc_info}.
        """
    info = (ValueError, ValueError('foo'), None)
    self.assertTrue(info is excInfoOrFailureToExcInfo(info))