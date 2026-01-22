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
def test_unsupportedMethods(self) -> None:
    """
        Some L{LoggingFile} methods are unsupported.
        """
    f = LoggingFile(self.logger)
    self.assertRaises(IOError, f.read)
    self.assertRaises(IOError, f.next)
    self.assertRaises(IOError, f.readline)
    self.assertRaises(IOError, f.readlines)
    self.assertRaises(IOError, f.xreadlines)
    self.assertRaises(IOError, f.seek)
    self.assertRaises(IOError, f.tell)
    self.assertRaises(IOError, f.truncate)