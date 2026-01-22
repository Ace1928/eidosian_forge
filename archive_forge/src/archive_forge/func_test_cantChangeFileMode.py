from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_cantChangeFileMode(self) -> None:
    """
        Opening a L{LogFile} which can be read and write but whose mode can't
        be changed doesn't trigger an error.
        """
    if runtime.platform.isWindows():
        name, directory = ('NUL', '')
        expectedPath = 'NUL'
    else:
        name, directory = ('null', '/dev')
        expectedPath = '/dev/null'
    log = logfile.LogFile(name, directory, defaultMode=365)
    self.addCleanup(log.close)
    self.assertEqual(log.path, expectedPath)
    self.assertEqual(log.defaultMode, 365)