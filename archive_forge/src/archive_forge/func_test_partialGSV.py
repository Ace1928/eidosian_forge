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
def test_partialGSV(self) -> None:
    """
        A partial GSV sentence is correctly parsed.
        """
    expected = {'type': 'GPGSV', 'GSVSentenceIndex': '3', 'numberOfGSVSentences': '3', 'numberOfSatellitesSeen': '11', 'azimuth_0': '067', 'azimuth_1': '311', 'azimuth_2': '244', 'elevation_0': '42', 'elevation_1': '14', 'elevation_2': '05', 'satellitePRN_0': '22', 'satellitePRN_1': '24', 'satellitePRN_2': '27', 'signalToNoiseRatio_0': '42', 'signalToNoiseRatio_1': '43', 'signalToNoiseRatio_2': '00'}
    self._parserTest(GPGSV_LAST, expected)