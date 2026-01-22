from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_namespaceDefault(self) -> None:
    """
        Default namespace is module name.
        """
    log = Logger()
    self.assertEqual(log.namespace, __name__)