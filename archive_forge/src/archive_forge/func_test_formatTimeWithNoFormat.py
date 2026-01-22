from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatTimeWithNoFormat(self) -> None:
    """
        If C{timeFormat} argument is L{None}, we get the default output.
        """
    t = mktime((2013, 9, 24, 11, 40, 47, 1, 267, -1))
    self.assertEqual(formatTime(t, timeFormat=None), '-')
    self.assertEqual(formatTime(t, timeFormat=None, default='!'), '!')