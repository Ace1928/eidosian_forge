import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def oriented(self):
    """
        Returns the one of {self, opposite} which is the *head* of the
        corresponding oriented edge of the link.
        """
    c, e = (self.crossing, self.strand_index)
    if c.sign == 1 and e in [0, 3] or (c.sign == -1 and e in [0, 1]):
        return self
    return self.opposite()