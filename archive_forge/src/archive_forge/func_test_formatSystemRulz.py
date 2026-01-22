from typing import AnyStr, Dict, Optional, cast
from twisted.python.failure import Failure
from twisted.python.test.test_tzhelper import addTZCleanup, mktime, setTZ
from twisted.trial import unittest
from twisted.trial.unittest import SkipTest
from .._format import (
from .._interfaces import LogEvent
from .._levels import LogLevel
def test_formatSystemRulz(self) -> None:
    """
        System is not supplanted by namespace and level.
        """
    event = dict(log_format='XYZZY', log_namespace='my.namespace', log_level=LogLevel.warn, log_system='S.Y.S.T.E.M.')
    self.assertEqual(formatEventAsClassicLogText(event), '- [S.Y.S.T.E.M.] XYZZY\n')