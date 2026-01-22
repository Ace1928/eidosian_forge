from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_easternHemisphere(self) -> None:
    """
        Positive longitudes are in the eastern hemisphere.
        """
    coordinate = base.Coordinate(1.0, Angles.LONGITUDE)
    self.assertEqual(coordinate.hemisphere, Directions.EAST)