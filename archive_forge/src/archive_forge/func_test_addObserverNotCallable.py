from typing import Dict, List, Tuple, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._logger import Logger
from .._observer import LogPublisher
def test_addObserverNotCallable(self) -> None:
    """
        L{LogPublisher.addObserver} refuses to add an observer that's
        not callable.
        """
    publisher = LogPublisher()
    self.assertRaises(TypeError, publisher.addObserver, object())