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
def build_one_skeleton(self):
    """
        Construct the 1-skeleton, i.e. record which edges are
        connected to which vertices.  This assumes that Edges and Vertices
        have already been built.
        """
    for edge in self.Edges:
        tet = edge.Corners[0].Tetrahedron
        one_subsimplex = edge.Corners[0].Subsimplex
        tail = tet.Class[Tail[one_subsimplex]]
        head = tet.Class[Head[one_subsimplex]]
        edge.Vertices = [tail, head]
        tail.Edges.append(edge)
        head.Edges.append(edge)
        if edge.IntOrBdry == 'bdry':
            tail.IntOrBdry = 'bdry'
            head.IntOrBdry = 'bdry'
    for vertex in self.Vertices:
        if vertex.IntOrBdry == '':
            vertex.IntOrBdry = 'int'