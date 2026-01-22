from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatTracebackHandlesUTF8DecodeFailure(self) -> None:
    """
        An error raised attempting to decode the UTF still produces a
        valid log message.
        """
    try:
        raise CapturedError(b'\xff\xfet\x00e\x00s\x00t\x00')
    except CapturedError:
        f = Failure()
    event: LogEvent = {'log_format': 'This is a test log message'}
    event['log_failure'] = f
    eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
    self.assertIn('Traceback', eventText)
    self.assertIn('CapturedError(b"\\xff\\xfet\\x00e\\x00s\\x00t\\x00")', eventText)