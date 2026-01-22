import errno
from functools import wraps
from os import getpid, name as SYSTEM_NAME
from typing import Any, Callable, Optional
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
import twisted.trial.unittest
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
from ...runner import _pidfile
from .._pidfile import (
@ifPlatformSupported
def test_isRunningInit(self) -> None:
    """
        L{PIDFile.isRunning} returns true for a process that we are not allowed
        to kill (errno=EPERM).

        @note: This differs from L{PIDFileTests.test_isRunningNotAllowed} in
        that it actually invokes the C{kill} system call, which is useful for
        testing of our chosen method for probing the existence of a process
        that we are not allowed to kill.

        @note: In this case, we try killing C{init}, which is process #1 on
        POSIX systems, so this test is not portable.  C{init} should always be
        running and should not be killable by non-root users.
        """
    if SYSTEM_NAME != 'posix':
        raise SkipTest('This test assumes POSIX')
    pidFile = PIDFile(self.filePath())
    pidFile._write(1)
    self.assertTrue(pidFile.isRunning())