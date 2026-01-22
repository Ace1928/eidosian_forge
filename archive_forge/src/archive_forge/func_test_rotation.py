from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_rotation(self) -> None:
    """
        Daily log files rotate daily.
        """
    log = RiggedDailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    days = [self.path + '.' + log.suffix(day * 86400) for day in range(3)]
    log._clock = 0.0
    log.write('123')
    log._clock = 43200
    log.write('4567890')
    log._clock = 86400
    log.write('1' * 11)
    self.assertTrue(os.path.exists(days[0]))
    self.assertFalse(os.path.exists(days[1]))
    log._clock = 172800
    log.write('')
    self.assertTrue(os.path.exists(days[0]))
    self.assertTrue(os.path.exists(days[1]))
    self.assertFalse(os.path.exists(days[2]))
    log._clock = 259199
    log.write('3')
    self.assertFalse(os.path.exists(days[2]))