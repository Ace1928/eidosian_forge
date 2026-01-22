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
def test_fixAltitude(self) -> None:
    """
        The NMEA representation of an altitude (above mean sea level)
        is correctly converted.
        """
    key, value = ('altitude', '545.4')
    altitude = base.Altitude(float(value))
    self._fixerTest({key: value}, _State(altitude=altitude))