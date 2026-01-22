from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_basicLogger(self) -> None:
    """
        Test that log levels and messages are emitted correctly for
        Logger.
        """
    log = TestLogger()
    for level in LogLevel.iterconstants():
        format = 'This is a {level_name} message'
        message = format.format(level_name=level.name)
        logMethod = getattr(log, level.name)
        logMethod(format, junk=message, level_name=level.name)
        self.assertEqual(log.emitted['level'], level)
        self.assertEqual(log.emitted['format'], format)
        self.assertEqual(log.emitted['kwargs']['junk'], message)
        self.assertTrue(hasattr(log, 'event'), 'No event observed.')
        self.assertEqual(log.event['log_format'], format)
        self.assertEqual(log.event['log_level'], level)
        self.assertEqual(log.event['log_namespace'], __name__)
        self.assertIsNone(log.event['log_source'])
        self.assertEqual(log.event['junk'], message)
        self.assertEqual(formatEvent(log.event), message)