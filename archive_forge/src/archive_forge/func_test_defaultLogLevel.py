from typing import Iterable, List, Tuple, Union, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from constantly import NamedConstant
from twisted.trial import unittest
from .._filter import (
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._observer import LogPublisher, bitbucketLogObserver
def test_defaultLogLevel(self) -> None:
    """
        Default log level is used.
        """
    predicate = LogLevelFilterPredicate()
    for default in ('', cast(str, None)):
        self.assertEqual(predicate.logLevelForNamespace(default), predicate.defaultLogLevel)
        self.assertEqual(predicate.logLevelForNamespace('rocker.cool.namespace'), predicate.defaultLogLevel)