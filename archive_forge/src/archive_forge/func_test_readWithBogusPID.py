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
def test_readWithBogusPID(self) -> None:
    """
        L{PIDFile.read} raises L{InvalidPIDFileError} when given an empty file
        path.
        """
    pidValue = b'$foo!'
    pidFile = PIDFile(self.filePath(pidValue))
    e = self.assertRaises(InvalidPIDFileError, pidFile.read)
    self.assertEqual(str(e), f'non-integer PID value in PID file: {pidValue!r}')