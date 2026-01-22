from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatFormatMultiLine(self) -> None:
    """
        If the formatted event has newlines, indent additional lines.
        """
    event = dict(log_format='XYZZY\nA hollow voice says:\n"Plugh"')
    self.assertEqual(formatEventAsClassicLogText(event), '- [-#-] XYZZY\n\tA hollow voice says:\n\t"Plugh"\n')