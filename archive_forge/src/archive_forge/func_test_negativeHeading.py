from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_negativeHeading(self) -> None:
    """
        Negative heading values raise C{ValueError}.
        """
    self._badValueTest(angleValue=-10.0)