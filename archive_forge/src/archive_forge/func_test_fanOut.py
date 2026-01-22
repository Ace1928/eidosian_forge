from typing import Dict, List, Tuple, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._logger import Logger
from .._observer import LogPublisher
def test_fanOut(self) -> None:
    """
        L{LogPublisher} calls its observers.
        """
    event = dict(foo=1, bar=2)
    events1: List[LogEvent] = []
    events2: List[LogEvent] = []
    events3: List[LogEvent] = []
    o1 = cast(ILogObserver, events1.append)
    o2 = cast(ILogObserver, events2.append)
    o3 = cast(ILogObserver, events3.append)
    publisher = LogPublisher(o1, o2, o3)
    publisher(event)
    self.assertIn(event, events1)
    self.assertIn(event, events2)
    self.assertIn(event, events3)