from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
def test_observeWrites(self) -> None:
    """
        L{FileLogObserver} writes to the given file when it observes events.
        """
    with StringIO() as fileHandle:
        observer = FileLogObserver(fileHandle, lambda e: str(e))
        event = dict(x=1)
        observer(event)
        self.assertEqual(fileHandle.getvalue(), str(event))