import io
from typing import IO, Any, List, Optional, TextIO, Tuple, Type, cast
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._file import textFileLogObserver
from .._global import MORE_THAN_ONCE_WARNING, LogBeginner
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
from ..test.test_stdlib import nextLine
def test_overrideBufferLimit(self) -> None:
    """
        The size of the L{LogBeginner} event buffer can be overridden with the
        C{initialBufferSize} initilizer argument.
        """
    limit = 3
    beginner = LogBeginner(self.publisher, self.errorStream, self.sysModule, self.warningsModule, initialBufferSize=limit)
    self._bufferLimitTest(limit, beginner)