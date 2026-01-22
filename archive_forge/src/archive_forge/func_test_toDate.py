from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_toDate(self) -> None:
    """
        Test that L{DailyLogFile.toDate} converts its timestamp argument to a
        time tuple (year, month, day).
        """
    log = logfile.DailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    timestamp = time.mktime((2000, 1, 1, 0, 0, 0, 0, 0, 0))
    self.assertEqual((2000, 1, 1), log.toDate(timestamp))