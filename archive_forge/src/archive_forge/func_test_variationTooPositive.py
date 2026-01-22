from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_variationTooPositive(self) -> None:
    """
        Variation values greater than C{180.0} raise C{ValueError}.
        """
    self._badValueTest(variationValue=190.0)