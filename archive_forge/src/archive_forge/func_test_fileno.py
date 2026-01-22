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
def test_fileno(self) -> None:
    """
        L{LoggingFile.fileno} returns C{-1}.
        """
    f = LoggingFile(self.logger)
    self.assertEqual(f.fileno(), -1)