from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_logInvalidLogLevel(self) -> None:
    """
        Test passing in a bogus log level to C{emit()}.
        """
    log = TestLogger()
    log.emit('*bogus*')
    errors = self.flushLoggedErrors(InvalidLogLevelError)
    self.assertEqual(len(errors), 1)