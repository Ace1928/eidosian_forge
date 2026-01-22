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
def test_readUTF8Bytes(self) -> None:
    """
        If the file being read from vends L{bytes}, strings decode from JSON as
        UTF-8.
        """
    with BytesIO(b'\x1e{"currency": "\xe2\x82\xac"}\n') as fileHandle:
        events = iter(eventsFromJSONLogFile(fileHandle))
        self.assertEqual(next(events), {'currency': '€'})
        self.assertRaises(StopIteration, next, events)
        self.assertEqual(len(self.errorEvents), 0)