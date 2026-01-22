import logging as py_logging
from time import time
from typing import List, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python import context, log as legacyLog
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._format import formatEvent
from .._interfaces import ILogObserver, LogEvent
from .._legacy import LegacyLogObserverWrapper, publishToNewObserver
from .._levels import LogLevel
def test_stdlibLogLevelWithString(self) -> None:
    """
        If the old-style C{"logLevel"} key is set to a standard library logging
        level, using a string value, the new-style C{"log_level"} key should
        get set to the corresponding log level.
        """
    publishToNewObserver(self.observer, self.legacyEvent(logLevel='WARNING'), legacyLog.textFromEventDict)
    self.assertEqual(self.events[0]['log_level'], LogLevel.warn)