from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_withoutInvariant(self) -> None:
    """
        L{base.PositionError}s can be instantiated with just a HDOP.
        """
    positionError = base.PositionError(hdop=1.0)
    self.assertEqual(positionError.hdop, 1.0)