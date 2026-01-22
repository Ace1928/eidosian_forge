import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def connected_sum(self, other_knot):
    """
        Returns the connected sum of two knots.

        >>> fig8 = [(1,7,2,6), (5,3,6,2), (7,4,0,5), (3,0,4,1)]
        >>> K = Link(fig8)
        >>> K.connected_sum(K)
        <Link: 1 comp; 8 cross>
        """
    first = self.copy()
    second = other_knot.copy()
    f1, i1 = (first.crossings[0], 0)
    f2, i2 = f1.adjacent[i1]
    g1, j1 = (second.crossings[0], 0)
    g2, j2 = g1.adjacent[j1]
    f1[i1] = g2[j2]
    f2[i2] = g1[j1]
    for c in first.crossings:
        c.label = (c.label, 1)
    for c in second.crossings:
        c.label = (c.label, 2)
        first.crossings.append(c)
    return type(self)(first.crossings)