from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_eventAsTextTimestampOnly(self) -> None:
    """
        If includeTimestamp is specified as the only option no system or
        traceback are printed.
        """
    if tzset is None:
        raise SkipTest('Platform cannot change timezone; unable to verify offsets.')
    addTZCleanup(self)
    setTZ('UTC+00')
    try:
        raise CapturedError('This is a fake error')
    except CapturedError:
        f = Failure()
    t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
    event: LogEvent = {'log_format': 'ABCD', 'log_system': 'fake_system', 'log_time': t}
    event['log_failure'] = f
    eventText = eventAsText(event, includeTimestamp=True, includeTraceback=False, includeSystem=False)
    self.assertEqual(eventText, '2013-09-24T11:40:47+0000 ABCD')