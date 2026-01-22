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
def test_emptyMiddleGSV(self) -> None:
    """
        A GSV sentence with empty entries in any position does not mean that
        entries in subsequent positions of the same GSV sentence are ignored.
        """
    sentences = [GPGSV_EMPTY_MIDDLE]
    callbacksFired = ['beaconInformationReceived']

    def checkBeaconInformation() -> None:
        beaconInformation = self.adapter._state['beaconInformation']
        seenBeacons = beaconInformation.seenBeacons
        self.assertEqual(len(seenBeacons), 2)
        self.assertIn(13, [b.identifier for b in seenBeacons])
    self._receiverTest(sentences, callbacksFired, checkBeaconInformation)