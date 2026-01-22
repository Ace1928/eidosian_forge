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
def test_beginLoggingToPreservesEncoding(self) -> None:
    """
        When L{LogBeginner.beginLoggingTo} redirects stdout/stderr streams, the
        replacement streams will preserve the encoding of the replaced streams,
        to minimally disrupt any application relying on a specific encoding.
        """
    weird = io.TextIOWrapper(io.BytesIO(), 'shift-JIS')
    weirderr = io.TextIOWrapper(io.BytesIO(), 'big5')
    self.sysModule.stdout = weird
    self.sysModule.stderr = weirderr
    events: List[LogEvent] = []
    self.beginner.beginLoggingTo([cast(ILogObserver, events.append)])
    stdout = cast(TextIO, self.sysModule.stdout)
    stderr = cast(TextIO, self.sysModule.stderr)
    self.assertEqual(stdout.encoding, 'shift-JIS')
    self.assertEqual(stderr.encoding, 'big5')
    stdout.write(b'\x97\x9b\n')
    stderr.write(b'\xbc\xfc\n')
    compareEvents(self, events, [dict(log_io='李'), dict(log_io='瑩')])