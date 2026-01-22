from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_noKillCall(self) -> None:
    """
        Verify that when L{lockfile.kill} does end up as None (e.g. on Windows
        without pywin32), it doesn't end up being called and raising a
        L{TypeError}.
        """
    self.patch(lockfile, 'kill', None)
    fl = lockfile.FilesystemLock(self.mktemp())
    fl.lock()
    self.assertFalse(fl.lock())