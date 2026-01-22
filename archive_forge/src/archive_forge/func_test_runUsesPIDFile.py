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
def test_runUsesPIDFile(self) -> None:
    """
        L{Runner.run} uses the provided PID file.
        """
    pidFile = DummyPIDFile()
    runner = Runner(reactor=MemoryReactor(), pidFile=pidFile)
    self.assertFalse(pidFile.entered)
    self.assertFalse(pidFile.exited)
    runner.run()
    self.assertTrue(pidFile.entered)
    self.assertTrue(pidFile.exited)