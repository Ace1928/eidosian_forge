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
def test_typicalGSA(self) -> None:
    """
        A typical GSA sentence is correctly parsed.
        """
    expected = {'type': 'GPGSA', 'dataMode': 'A', 'fixType': '3', 'usedSatellitePRN_0': '19', 'usedSatellitePRN_1': '28', 'usedSatellitePRN_2': '14', 'usedSatellitePRN_3': '18', 'usedSatellitePRN_4': '27', 'usedSatellitePRN_5': '22', 'usedSatellitePRN_6': '31', 'usedSatellitePRN_7': '39', 'positionDilutionOfPrecision': '1.7', 'horizontalDilutionOfPrecision': '1.0', 'verticalDilutionOfPrecision': '1.3'}
    self._parserTest(GPGSA, expected)