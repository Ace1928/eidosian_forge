from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def correctedHeading(self):
    """
        Corrects the heading by the given variation. This is sometimes known as
        the true heading.

        @return: The heading, corrected by the variation. If the variation or
            the angle are unknown, returns L{None}.
        @rtype: C{float} or L{None}
        """
    if self._angle is None or self.variation is None:
        return None
    angle = (self.inDecimalDegrees - self.variation.inDecimalDegrees) % 360
    return Angle(angle, Angles.HEADING)