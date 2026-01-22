from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatUnformattableEventWithUnformattableValue(self) -> None:
    """
        Formatting an unformattable event that has an unformattable value.
        """
    event = dict(log_format='{evil()}', evil=lambda: 1 / 0, gurk=Unformattable())
    result = formatEvent(event)
    self.assertIn('MESSAGE LOST: unformattable object logged:', result)
    self.assertIn('Recoverable data:', result)
    self.assertIn('Exception during formatting:', result)