from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
def test_withHeading(self) -> None:
    """
        Variation values get combined with headings correctly.
        """
    trueHeading, variation, direction = ('123.12', '1.34', 'E')
    sentenceData = {'trueHeading': trueHeading, 'magneticVariation': variation, 'magneticVariationDirection': direction}
    heading = base.Heading.fromFloats(float(trueHeading), variationValue=float(variation))
    self._fixerTest(sentenceData, {'heading': heading})