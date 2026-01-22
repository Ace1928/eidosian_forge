from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_lockCleanedUpConcurrently(self) -> None:
    """
        If a second process cleans up the lock after a first one checks the
        lock and finds that no process is holding it, the first process does
        not fail when it tries to clean up the lock.
        """

    def fakeRmlink(name: str) -> None:
        rmlinkPatch.restore()
        lockfile.rmlink(lockf)
        return lockfile.rmlink(name)
    rmlinkPatch = self.patch(lockfile, 'rmlink', fakeRmlink)

    def fakeKill(pid: int, signal: int) -> None:
        if signal != 0:
            raise OSError(errno.EPERM, None)
        if pid == 43125:
            raise OSError(errno.ESRCH, None)
    self.patch(lockfile, 'kill', fakeKill)
    lockf = self.mktemp()
    lock = lockfile.FilesystemLock(lockf)
    lockfile.symlink(str(43125), lockf)
    self.assertTrue(lock.lock())
    self.assertTrue(lock.clean)
    self.assertTrue(lock.locked)