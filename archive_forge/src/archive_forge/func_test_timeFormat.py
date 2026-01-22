from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
def test_timeFormat(self) -> None:
    """
        Returned L{FileLogObserver} has the correct outFile.
        """
    with StringIO() as fileHandle:
        observer = textFileLogObserver(fileHandle, timeFormat='%f')
        observer(dict(log_format='XYZZY', log_time=112345.6))
        self.assertEqual(fileHandle.getvalue(), '600000 [-#-] XYZZY\n')