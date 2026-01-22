from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def inFeet(self):
    """
        Gets the altitude this object represents, in feet.

        @return: The altitude, expressed in feet.
        @rtype: C{float}
        """
    return self._altitude / METERS_PER_FOOT