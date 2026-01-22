from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_inDegreesMinutesSeconds(self) -> None:
    """
        Coordinate values can be accessed in degrees, minutes, seconds.
        """
    c = base.Coordinate(50.5, Angles.LATITUDE)
    self.assertEqual(c.inDegreesMinutesSeconds, (50, 30, 0))
    c = base.Coordinate(50.213, Angles.LATITUDE)
    self.assertEqual(c.inDegreesMinutesSeconds, (50, 12, 46))