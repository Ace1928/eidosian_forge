from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_readlinkError(self) -> None:
    """
        An exception raised by C{readlink} other than C{ENOENT} is passed up to
        the caller of L{FilesystemLock.lock}.
        """
    self._readlinkErrorTest(OSError, errno.ENOSYS)
    self._readlinkErrorTest(IOError, errno.ENOSYS)