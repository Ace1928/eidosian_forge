from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_abstractShouldRotate(self) -> None:
    """
        L{BaseLogFile.shouldRotate} is abstract and must be implemented by
        subclass.
        """
    log = logfile.BaseLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    self.assertRaises(NotImplementedError, log.shouldRotate)