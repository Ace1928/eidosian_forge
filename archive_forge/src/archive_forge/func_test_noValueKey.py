from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
def test_noValueKey(self) -> None:
    """
        Tests that when no C{valueKey} is provided, C{unitKey} is used, minus
        C{"Units"} at the end.
        """

    class FakeSentence:
        """
            A fake sentence that just has a "foo" attribute.
            """

        def __init__(self) -> None:
            self.foo = 1
    self.adapter.currentSentence = FakeSentence()
    self.adapter._fixUnits(unitKey='fooUnits', unit='N')
    self.assertNotEqual(self.adapter._sentenceData['foo'], 1)