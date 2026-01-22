from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_headingWithoutVariationRepr(self) -> None:
    """
        A repr of a heading with no variation reports its value and that the
        variation is unknown.
        """
    heading = base.Heading(1.0)
    expectedRepr = '<Heading (1.0 degrees, unknown variation)>'
    self.assertEqual(repr(heading), expectedRepr)