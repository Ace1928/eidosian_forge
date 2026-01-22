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
def blowup2(self, n):
    """
        Create ``n`` edges of valence 2 in random places, removing valence
        3 edges whenever they appear.
        """
    for i in range(n):
        rand_edge = self.Edges[random.randint(0, len(self.Edges) - 1)]
        j = random.randint(0, len(rand_edge.Corners) - 1)
        k = random.randint(0, len(rand_edge.Corners) - 1 - j)
        one_subsimplex = rand_edge.Corners[j].Subsimplex
        two_subsimplex = LeftFace[one_subsimplex]
        a = Arrow(one_subsimplex, two_subsimplex, rand_edge.Corners[j].Tetrahedron)
        self.zero_to_two(a, k)
        self.eliminate_valence_three()
    return len(self)