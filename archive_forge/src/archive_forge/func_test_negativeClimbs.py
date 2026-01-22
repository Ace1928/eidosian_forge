from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_negativeClimbs(self) -> None:
    """
        Climbs can have negative values, and still report that value
        in meters per second and when converted to floats.
        """
    climb = base.Climb(-42.0)
    self.assertEqual(climb.inMetersPerSecond, -42.0)
    self.assertEqual(float(climb), -42.0)