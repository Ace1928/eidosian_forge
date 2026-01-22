from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_cleanlyAcquire(self) -> None:
    """
        If the lock has never been held, it can be acquired and the C{clean}
        and C{locked} attributes are set to C{True}.
        """
    lockf = self.mktemp()
    lock = lockfile.FilesystemLock(lockf)
    self.assertTrue(lock.lock())
    self.assertTrue(lock.clean)
    self.assertTrue(lock.locked)