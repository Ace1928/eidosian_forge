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
def test_isRunningNoPIDFile(self) -> None:
    """
        L{PIDFile.isRunning} returns false if the PID file doesn't exist.
        """
    pidFile = PIDFile(self.filePath())
    self.assertFalse(pidFile.isRunning())