from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_eventWithTraceback(self) -> None:
    """
        An event with a C{log_failure} key will have a traceback appended.
        """
    try:
        raise CapturedError('This is a fake error')
    except CapturedError:
        f = Failure()
    event: LogEvent = {'log_format': 'This is a test log message'}
    event['log_failure'] = f
    eventText = eventAsText(event, includeTimestamp=True, includeSystem=False)
    self.assertIn(str(f.getTraceback()), eventText)
    self.assertIn('This is a test log message', eventText)