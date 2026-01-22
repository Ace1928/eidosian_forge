from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatUnformattableEvent(self) -> None:
    """
        Formatting an event that's just plain out to get us.
        """
    event = dict(log_format='{evil()}', evil=lambda: 1 / 0)
    result = formatEvent(event)
    self.assertIn('Unable to format event', result)
    self.assertIn(repr(event), result)