from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_correctedHeadingUnderflowEdgeCase(self) -> None:
    """
        A heading with a value and a variation has the appropriate corrected
        heading value, even when the variation puts it exactly at the 0
        degree boundary.
        """
    h = base.Heading.fromFloats(1.0, variationValue=1.0)
    self.assertEqual(h.correctedHeading, base.Angle(0.0, Angles.HEADING))