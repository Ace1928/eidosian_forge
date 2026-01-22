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
def test_isRunningDoesExist(self) -> None:
    """
        L{PIDFile.isRunning} returns true for a process that does exist.
        """
    pidFile = PIDFile(self.filePath())
    pidFile._write(1337)

    def kill(pid: int, signal: int) -> None:
        return
    self.patch(_pidfile, 'kill', kill)
    self.assertTrue(pidFile.isRunning())