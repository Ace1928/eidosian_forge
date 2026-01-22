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
def build_vertex_incidences(self):
    for vertex in self.Vertices:
        vertex.IncidenceVector = linalg.Vector(4 * len(self))
        for corner in vertex.Corners:
            j = corner.Tetrahedron.Index
            vertex.IncidenceVector[4 * j:4 * j + 4] += VertexVector[corner.Subsimplex]