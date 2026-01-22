from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatTimeDefault(self) -> None:
    """
        Time is first field.  Default time stamp format is RFC 3339 and offset
        respects the timezone as set by the standard C{TZ} environment variable
        and L{tzset} API.
        """
    if tzset is None:
        raise SkipTest('Platform cannot change timezone; unable to verify offsets.')
    addTZCleanup(self)
    setTZ('UTC+00')
    t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
    event = dict(log_format='XYZZY', log_time=t)
    self.assertEqual(formatEventAsClassicLogText(event), '2013-09-24T11:40:47+0000 [-#-] XYZZY\n')