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
def forwardAndVerify(self, event: LogEvent) -> LogEvent:
    """
        Send an event to a wrapped legacy observer and verify that its data is
        preserved.

        @param event: an event

        @return: the event as observed by the legacy wrapper
        """
    event.setdefault('log_time', time())
    event.setdefault('log_system', '-')
    event.setdefault('log_level', LogLevel.info)
    observed = self.observe(dict(event))
    for key, value in event.items():
        self.assertIn(key, observed)
    return observed