from .simplex import *
from .corner import Corner
from .arrow import Arrow
from .perm4 import Perm4
import sys
def get_arrow(self):
    e = self.Corners[0].Subsimplex
    return Arrow(e, RightFace[e], self.Corners[0].Tetrahedron)