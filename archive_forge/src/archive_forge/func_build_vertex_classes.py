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
def build_vertex_classes(self):
    for tet in self.Tetrahedra:
        for zero_subsimplex in ZeroSubsimplices:
            if tet.Class[zero_subsimplex] is None:
                newVertex = Vertex()
                self.Vertices.append(newVertex)
                self.walk_vertex(newVertex, zero_subsimplex, tet)
    for i in range(len(self.Vertices)):
        self.Vertices[i].Index = i