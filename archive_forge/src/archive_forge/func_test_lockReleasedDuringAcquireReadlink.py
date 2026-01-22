from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipUnless(platform.isWindows(), 'special readlink EACCES handling only necessary and correct on Windows.')
def test_lockReleasedDuringAcquireReadlink(self) -> None:
    """
        If the lock is initially held but is released while an attempt
        is made to acquire it, the lock attempt fails and
        L{FilesystemLock.lock} returns C{False}.
        """

    def fakeReadlink(name: str) -> NoReturn:
        raise OSError(errno.EACCES, None)
    self.patch(lockfile, 'readlink', fakeReadlink)
    lockf = self.mktemp()
    lock = lockfile.FilesystemLock(lockf)
    lockfile.symlink(str(43125), lockf)
    self.assertFalse(lock.lock())
    self.assertFalse(lock.locked)