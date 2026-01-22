from .simplex import *
from .corner import Corner
from .arrow import Arrow
from .perm4 import Perm4
import sys
def orientation_with_respect_to(self, tet, a, b):
    try:
        return self._edge_orient_cache[tet, a, b]
    except IndexError:
        raise ValueError('Given corner of tet not on this edge')