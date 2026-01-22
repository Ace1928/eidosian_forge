from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_headingWithVariationRepr(self) -> None:
    """
        A repr of a heading with known variation reports its value and the
        value of that variation.
        """
    angle, variation = (1.0, -10.0)
    heading = base.Heading.fromFloats(angle, variationValue=variation)
    reprTemplate = '<Heading ({0} degrees, <Variation ({1} degrees)>)>'
    self.assertEqual(repr(heading), reprTemplate.format(angle, variation))