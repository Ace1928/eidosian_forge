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
def test_levelNotChange(self) -> None:
    """
        If explicitly set, the C{isError} key will be preserved when forwarding
        from a new-style logging emitter to a legacy logging observer,
        regardless of log level.
        """
    self.forwardAndVerify(dict(log_level=LogLevel.info, isError=1))
    self.forwardAndVerify(dict(log_level=LogLevel.warn, isError=1))
    self.forwardAndVerify(dict(log_level=LogLevel.error, isError=0))
    self.forwardAndVerify(dict(log_level=LogLevel.critical, isError=0))