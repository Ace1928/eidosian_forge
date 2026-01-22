import errno
from io import StringIO
from signal import SIGTERM
from types import TracebackType
from typing import Any, Iterable, List, Optional, TextIO, Tuple, Type, Union, cast
from attr import Factory, attrib, attrs
import twisted.trial.unittest
from twisted.internet.testing import MemoryReactor
from twisted.logger import (
from twisted.python.filepath import FilePath
from ...runner import _runner
from .._exit import ExitStatus
from .._pidfile import NonePIDFile, PIDFile
from .._runner import Runner
def test_killRequestedWithPIDFileCantRead(self) -> None:
    """
        L{Runner.killIfRequested} when C{kill} is true and given a C{pidFile}
        that it can't read exits with L{ExitStatus.EX_IOERR}.
        """
    pidFile = PIDFile(self.filePath(None))

    def read() -> int:
        raise OSError(errno.EACCES, 'Permission denied')
    pidFile.read = read
    runner = Runner(reactor=MemoryReactor(), kill=True, pidFile=pidFile)
    runner.killIfRequested()
    self.assertEqual(self.exit.status, ExitStatus.EX_IOERR)
    self.assertEqual(self.exit.message, 'Unable to read PID file.')