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
def legacyEvent(self, *message: str, **values: object) -> legacyLog.EventDict:
    """
        Return a basic old-style event as would be created by L{legacyLog.msg}.

        @param message: a message event value in the legacy event format
        @param values: additional event values in the legacy event format

        @return: a legacy event
        """
    event = (context.get(legacyLog.ILogContext) or {}).copy()
    event.update(values)
    event['message'] = message
    event['time'] = time()
    if 'isError' not in event:
        event['isError'] = 0
    return event