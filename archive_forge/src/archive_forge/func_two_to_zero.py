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
def two_to_zero(self, edge_or_arrow, must_succeed=False, unsafe_mode=False):
    """
        Flatten the star of an edge of valence 2 to eliminate two
        tetrahedra.

        Options and return value are the same as ``two_to_three``.
        """
    edge, a = edge_and_arrow(edge_or_arrow)
    b = a.glued()
    possible, reason = self._arrow_permits_two_to_zero(a)
    if not possible:
        if must_succeed:
            raise ValueError(reason)
        return False
    self._two_to_zero_hook(a)
    a.opposite().glued().reverse().glue(b.opposite().glued())
    a.reverse().glued().reverse().glue(b.reverse().glued())
    for corner in edge.Corners:
        self.delete_tet(corner.Tetrahedron)
    if not unsafe_mode:
        self.build_edge_classes()
        if VERBOSE:
            print('2->0')
            print(self.EdgeValences)
    return True