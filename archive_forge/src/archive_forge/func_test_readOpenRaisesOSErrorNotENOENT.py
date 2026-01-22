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
def test_readOpenRaisesOSErrorNotENOENT(self) -> None:
    """
        L{PIDFile.read} re-raises L{OSError} if the associated C{errno} is
        anything other than L{errno.ENOENT}.
        """

    def oops(mode: str='r') -> NoReturn:
        raise OSError(errno.EIO, 'I/O error')
    self.patch(FilePath, 'open', oops)
    pidFile = PIDFile(self.filePath())
    error = self.assertRaises(OSError, pidFile.read)
    self.assertEqual(error.errno, errno.EIO)