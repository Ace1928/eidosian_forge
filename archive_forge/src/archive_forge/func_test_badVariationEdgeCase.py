from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_badVariationEdgeCase(self) -> None:
    """
        Headings can not be instantiated with a variation of -180 degrees.
        """
    self._badValueTest(variationValue=-180.0)