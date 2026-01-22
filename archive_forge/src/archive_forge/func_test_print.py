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
def test_print(self) -> None:
    """
        L{LoggingFile} can replace L{sys.stdout}.
        """
    f = self.observedFile()
    self.patch(sys, 'stdout', f)
    print('Hello,', end=' ')
    print('world.')
    self.assertEqual(f.messages, ['Hello, world.'])