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
def test_fullGLL(self) -> None:
    """
        A full GLL sentence is correctly parsed.
        """
    expected = {'type': 'GPGLL', 'latitudeFloat': '4916.45', 'latitudeHemisphere': 'N', 'longitudeFloat': '12311.12', 'longitudeHemisphere': 'W', 'timestamp': '225444', 'dataMode': 'A'}
    self._parserTest(GPGLL, expected)