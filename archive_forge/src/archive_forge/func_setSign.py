from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
def setSign(self, sign):
    """
        Sets the sign of the variation of this heading.

        @param sign: The new sign. C{1} for positive and C{-1} for negative
            signs, respectively.
        @type sign: C{int}

        @raise ValueError: If the C{sign} parameter is not C{-1} or C{1}.
        """
    if self.variation.inDecimalDegrees is None:
        raise ValueError("can't set the sign of an unknown variation")
    self.variation.setSign(sign)