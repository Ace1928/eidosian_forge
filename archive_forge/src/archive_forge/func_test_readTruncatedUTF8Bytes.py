from io import BytesIO, StringIO
from typing import IO, Any, List, Optional, Sequence, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._flatten import extractField
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._json import (
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
def test_readTruncatedUTF8Bytes(self) -> None:
    """
        If the JSON text for a record is truncated in the middle of a two-byte
        Unicode codepoint, we don't want to see a codec exception and the
        stream is read properly when the additional data arrives.
        """
    with BytesIO(b'\x1e{"x": "\xe2\x82\xac"}\n') as fileHandle:
        events = iter(eventsFromJSONLogFile(fileHandle, bufferSize=8))
        self.assertEqual(next(events), {'x': '€'})
        self.assertRaises(StopIteration, next, events)
        self.assertEqual(len(self.errorEvents), 0)