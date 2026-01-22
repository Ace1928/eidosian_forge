from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_rotateAlreadyExists(self) -> None:
    """
        L{DailyLogFile.rotate} doesn't do anything if they new log file already
        exists on the disk.
        """
    log = RiggedDailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    newFilePath = f'{log.path}.{log.suffix(log.lastDate)}'
    with open(newFilePath, 'w') as fp:
        fp.write('123')
    previousFile = log._file
    log.rotate()
    self.assertEqual(previousFile, log._file)