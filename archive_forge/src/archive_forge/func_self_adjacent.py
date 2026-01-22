from .simplex import *
from .corner import Corner
from .arrow import Arrow
from .perm4 import Perm4
import sys
def self_adjacent(self):
    for corner in self.Corners:
        for one_subsimplex in AdjacentEdges[corner.Subsimplex]:
            if corner.Tetrahedron.Class[one_subsimplex] is self:
                return 1
    return 0