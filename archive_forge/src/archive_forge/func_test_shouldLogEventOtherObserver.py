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
def test_shouldLogEventOtherObserver(self) -> None:
    """
        Filtered results get sent to the other observer, if passed.
        """
    self.assertEqual(self.filterWith(['notTwo'], True), ([0, 1, 3], [2]))