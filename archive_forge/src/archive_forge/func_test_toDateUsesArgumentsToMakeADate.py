from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_toDateUsesArgumentsToMakeADate(self) -> None:
    """
        Test that L{DailyLogFile.toDate} uses its arguments to create a new
        date.
        """
    log = logfile.DailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    date = (2014, 10, 22)
    seconds = time.mktime(date + (0,) * 6)
    logDate = log.toDate(seconds)
    self.assertEqual(date, logDate)