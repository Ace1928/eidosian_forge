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
def smash_star(self, edge):
    """
        If an edge joins distinct vertices and has an embedded open
        star then the following method will smash each 3-simplex in
        the star down to a 2-simplex, and smash the edge to a vertex,
        reducing the number of vertices by 1.  Returns ``True`` on
        success, ``False`` on failure.
        """
    if not edge.distinct() or edge.Vertices[0] == edge.Vertices[1]:
        return False
    start = edge.get_arrow()
    a = start.copy()
    garbage = []
    while 1:
        garbage.append(a.Tetrahedron)
        top = a.opposite().glued()
        bottom = a.reverse().glued().reverse()
        bottom.glue(top)
        a.reverse().opposite().next()
        if a == start:
            break
    for tet in garbage:
        self.delete_tet(tet)
    self.rebuild()
    return True