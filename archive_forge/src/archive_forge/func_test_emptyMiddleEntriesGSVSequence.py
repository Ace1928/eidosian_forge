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
def test_emptyMiddleEntriesGSVSequence(self) -> None:
    """
        A complete sequence of GSV sentences with empty entries in the
        middle still fires C{beaconInformationReceived}.
        """
    sentences = [GPGSV_EMPTY_MIDDLE]
    self._receiverTest(sentences, ['beaconInformationReceived'])