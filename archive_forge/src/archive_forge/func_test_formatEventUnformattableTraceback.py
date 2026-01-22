from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatEventUnformattableTraceback(self) -> None:
    """
        If a traceback cannot be appended, a message indicating this is true
        is appended.
        """
    event: LogEvent = {'log_format': ''}
    event['log_failure'] = object()
    eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
    self.assertIsInstance(eventText, str)
    self.assertIn('(UNABLE TO OBTAIN TRACEBACK FROM EVENT)', eventText)