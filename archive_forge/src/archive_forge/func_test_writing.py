from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_writing(self) -> None:
    """
        A daily log file can be written to like an ordinary log file.
        """
    with contextlib.closing(RiggedDailyLogFile(self.name, self.dir)) as log:
        log.write('123')
        log.write('456')
        log.flush()
        log.write('7890')
    with open(self.path) as f:
        self.assertEqual(f.read(), '1234567890')