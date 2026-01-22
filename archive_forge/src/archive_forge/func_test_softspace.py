import sys
from typing import List, Optional
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._io import LoggingFile
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
def test_softspace(self) -> None:
    """
        L{LoggingFile.softspace} is 0.
        """
    self.assertEqual(LoggingFile(self.logger).softspace, 0)
    warningsShown = self.flushWarnings([self.test_softspace])
    self.assertEqual(len(warningsShown), 1)
    self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
    deprecatedClass = 'twisted.logger._io.LoggingFile.softspace'
    self.assertEqual(warningsShown[0]['message'], '%s was deprecated in Twisted 21.2.0' % deprecatedClass)