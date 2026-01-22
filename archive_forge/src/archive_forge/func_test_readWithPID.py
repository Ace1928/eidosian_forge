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
def test_readWithPID(self) -> None:
    """
        L{PIDFile.read} returns the PID from the given file path.
        """
    pid = 1337
    pidFile = PIDFile(self.filePath(PIDFile._format(pid=pid)))
    self.assertEqual(pid, pidFile.read())