from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_setBadVariationSign(self) -> None:
    """
        Setting the sign of a heading to values that aren't C{-1} or C{1}
        raises C{ValueError} and does not affect the heading.
        """
    h = base.Heading.fromFloats(1.0, variationValue=1.0)
    self.assertRaises(ValueError, h.setSign, -50)
    self.assertEqual(h.variation.inDecimalDegrees, 1.0)
    self.assertRaises(ValueError, h.setSign, 0)
    self.assertEqual(h.variation.inDecimalDegrees, 1.0)
    self.assertRaises(ValueError, h.setSign, 50)
    self.assertEqual(h.variation.inDecimalDegrees, 1.0)