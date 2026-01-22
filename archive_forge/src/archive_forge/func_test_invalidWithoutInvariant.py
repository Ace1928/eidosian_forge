from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_invalidWithoutInvariant(self) -> None:
    """
        Creating a L{base.PositionError} with values set to an impossible
        combination works if the invariant is not checked.
        """
    error = base.PositionError(pdop=1.0, vdop=1.0, hdop=1.0)
    self.assertEqual(error.pdop, 1.0)
    self.assertEqual(error.hdop, 1.0)
    self.assertEqual(error.vdop, 1.0)