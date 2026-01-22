from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_correctedHeading(self) -> None:
    """
        A heading with a value and a variation has a corrected heading.
        """
    h = base.Heading.fromFloats(1.0, variationValue=-10.0)
    self.assertEqual(h.correctedHeading, base.Angle(11.0, Angles.HEADING))