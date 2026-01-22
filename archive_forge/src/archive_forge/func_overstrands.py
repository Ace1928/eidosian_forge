import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def overstrands(self):
    """
        Returns a list of the sequences of overcrossings (which are lists of
        CrossingEntryPoints), sorted in descending order of length.

        >>> L = Link('L14n1000')
        >>> len(L.overstrands()[0])
        3
        """
    from . import simplify
    return simplify.over_or_under_strands(self, 'over')