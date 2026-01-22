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
def test_singleSentenceGSVSequence(self) -> None:
    """
        The parser does not fail badly when the sequence consists of
        only one sentence (but is otherwise complete).
        """
    sentences = [GPGSV_SINGLE]
    self._receiverTest(sentences, ['beaconInformationReceived'])