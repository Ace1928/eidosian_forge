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
def test_mixing(self) -> None:
    pdop, hdop, vdop = ('1', '1', '1')
    positionError = base.PositionError(pdop=float(pdop), hdop=float(hdop), vdop=float(vdop))
    sentenceData = {'positionDilutionOfPrecision': pdop, 'horizontalDilutionOfPrecision': hdop, 'verticalDilutionOfPrecision': vdop}
    self._fixerTest(sentenceData, {'positionError': positionError})