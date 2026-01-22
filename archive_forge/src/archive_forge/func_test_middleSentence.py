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
def test_middleSentence(self) -> None:
    """
        A sentence in the middle of a GSV sequence is correctly
        identified (as being neither the last nor the first).
        """
    self.protocol.lineReceived(GPGSV_MIDDLE)
    sentence = self.receiver.receivedSentence
    assert sentence is not None
    self.assertFalse(sentence._isFirstGSVSentence())
    self.assertFalse(sentence._isLastGSVSentence())