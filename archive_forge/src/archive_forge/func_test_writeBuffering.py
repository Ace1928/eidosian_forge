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
def test_writeBuffering(self) -> None:
    """
        Writing buffers correctly.
        """
    f = self.observedFile()
    f.write('Hello')
    self.assertEqual(f.messages, [])
    f.write(', world!\n')
    self.assertEqual(f.messages, ['Hello, world!'])
    f.write("It's nice to meet you.\n\nIndeed.")
    self.assertEqual(f.messages, ['Hello, world!', "It's nice to meet you.", ''])