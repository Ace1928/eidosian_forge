import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def crossing_entries(self):
    ans = []
    for C in self.crossings:
        ans += C.entry_points()
    return ans