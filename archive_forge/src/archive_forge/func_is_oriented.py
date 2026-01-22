from .simplex import *
from .tetrahedron import Tetrahedron
from .corner import Corner
from .arrow import Arrow
from .face import Face
from .edge import Edge
from .vertex import Vertex
from .surface import Surface, SpunSurface, ClosedSurface, ClosedSurfaceInCusped
from .perm4 import Perm4, inv
from . import files
from . import linalg
from . import homology
import sys
import random
import io
def is_oriented(self):
    for tet in self.Tetrahedra:
        for two_subsimplex in TwoSubsimplices:
            if not tet.Neighbor[two_subsimplex] is None and tet.Gluing[two_subsimplex].sign() == 0:
                return False
    return True