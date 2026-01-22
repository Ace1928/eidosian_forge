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
def test_beginLoggingToRedirectStandardIO(self) -> None:
    """
        L{LogBeginner.beginLoggingTo} will re-direct the standard output and
        error streams by setting the C{stdio} and C{stderr} attributes on its
        sys module object.
        """
    events: List[LogEvent] = []
    self.beginner.beginLoggingTo([cast(ILogObserver, events.append)])
    print('Hello, world.', file=cast(TextIO, self.sysModule.stdout))
    compareEvents(self, events, [dict(log_namespace='stdout', log_io='Hello, world.')])
    del events[:]
    print('Error, world.', file=cast(TextIO, self.sysModule.stderr))
    compareEvents(self, events, [dict(log_namespace='stderr', log_io='Error, world.')])