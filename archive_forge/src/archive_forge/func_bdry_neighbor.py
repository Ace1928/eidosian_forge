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
def bdry_neighbor(self, arrow):
    """
        Find a boundary face adjoining a given boundary face.
        Given an Arrow representing a boundary face, return the Arrow
        representing the boundary face that shares the Arrow's Edge.
        """
    if arrow.next() is not None:
        raise Insanity('That boundary face is not on the boundary!')
    edge = arrow.Tetrahedron.Class[arrow.Edge]
    if edge.LeftBdryArrow == arrow:
        return edge.RightBdryArrow
    else:
        return edge.LeftBdryArrow