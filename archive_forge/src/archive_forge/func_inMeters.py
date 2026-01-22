from functools import partial
from operator import attrgetter
from typing import ClassVar, Sequence
from zope.interface import implementer
from constantly import NamedConstant, Names
from twisted.positioning import ipositioning
from twisted.python.util import FancyEqMixin
@property
def inMeters(self):
    """
        Returns the altitude this object represents, in meters.

        @return: The altitude, expressed in feet.
        @rtype: C{float}
        """
    return self._altitude