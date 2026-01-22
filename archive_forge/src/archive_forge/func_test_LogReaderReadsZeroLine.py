from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_LogReaderReadsZeroLine(self) -> None:
    """
        L{LogReader.readLines} supports reading no line.
        """
    with open(self.path, 'w'):
        pass
    reader = logfile.LogReader(self.path)
    self.addCleanup(reader.close)
    self.assertEqual([], reader.readLines(0))