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
def test_beginLoggingToTwice(self) -> None:
    """
        When invoked twice, L{LogBeginner.beginLoggingTo} will emit a log
        message warning the user that they previously began logging, and add
        the new log observers.
        """
    events1: List[LogEvent] = []
    events2: List[LogEvent] = []
    fileHandle = io.StringIO()
    textObserver = textFileLogObserver(fileHandle)
    self.publisher(dict(event='prebuffer'))
    firstFilename, firstLine = nextLine()
    self.beginner.beginLoggingTo([cast(ILogObserver, events1.append), textObserver])
    self.publisher(dict(event='postbuffer'))
    secondFilename, secondLine = nextLine()
    self.beginner.beginLoggingTo([cast(ILogObserver, events2.append), textObserver])
    self.publisher(dict(event='postwarn'))
    warning = dict(log_format=MORE_THAN_ONCE_WARNING, log_level=LogLevel.warn, fileNow=secondFilename, lineNow=secondLine, fileThen=firstFilename, lineThen=firstLine)
    self.maxDiff = None
    compareEvents(self, events1, [dict(event='prebuffer'), dict(event='postbuffer'), warning, dict(event='postwarn')])
    compareEvents(self, events2, [warning, dict(event='postwarn')])
    output = fileHandle.getvalue()
    self.assertIn(f'<{firstFilename}:{firstLine}>', output)
    self.assertIn(f'<{secondFilename}:{secondLine}>', output)