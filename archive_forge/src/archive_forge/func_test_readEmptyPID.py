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
def test_readEmptyPID(self) -> None:
    """
        L{PIDFile.read} raises L{InvalidPIDFileError} when given an empty file
        path.
        """
    pidValue = b''
    pidFile = PIDFile(self.filePath(b''))
    e = self.assertRaises(InvalidPIDFileError, pidFile.read)
    self.assertEqual(str(e), f'non-integer PID value in PID file: {pidValue!r}')