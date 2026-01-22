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
def walk_vertex(self, vertex, zero_subsimplex, tet):
    if tet.Class[zero_subsimplex] is not None:
        return
    else:
        tet.Class[zero_subsimplex] = vertex
        vertex.Corners.append(Corner(tet, zero_subsimplex))
        for two_subsimplex in TwoSubsimplices:
            if is_subset(zero_subsimplex, two_subsimplex) and tet.Gluing[two_subsimplex] is not None:
                self.walk_vertex(vertex, tet.Gluing[two_subsimplex].image(zero_subsimplex), tet.Neighbor[two_subsimplex])