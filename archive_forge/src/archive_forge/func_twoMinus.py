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
@staticmethod
def twoMinus(event: LogEvent) -> NamedConstant:
    """
                count <= 2

                @param event: an event

                @return: L{PredicateResult.yes} if C{event["count"] <= 2},
                    otherwise L{PredicateResult.maybe}.
                """
    if event['count'] <= 2:
        return PredicateResult.yes
    return PredicateResult.maybe