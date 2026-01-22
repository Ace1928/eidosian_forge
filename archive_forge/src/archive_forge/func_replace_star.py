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
def replace_star(self, arrow, top_arrows, bottom_arrows):
    """
        This method takes an arrow and replaces its star with
        other_complex attaching that complex via top_arrows and
        bottom_arrows where: Let a be the arrow defining the same
        directed edge as arrow which is the ith such arrow counting
        around the star.  Then a.glued() is glued to top_arrow[i] and
        a.reverse().glued() is glued to bottom_arrow[i].

        NOTE:  If it fails, you need to delete any tets that you were
        trying to add.
        """
    edge = arrow.Tetrahedron.Class[arrow.Edge]
    a = arrow.copy().opposite()
    if not edge.IntOrBdry == 'int':
        return None
    if not edge.distinct():
        return None
    valence = edge.valence()
    if len(top_arrows) != valence or len(bottom_arrows) != valence:
        return None
    for i in range(valence):
        top_arrows[i].glue(a.glued())
        a.reverse()
        bottom_arrows[i].glue(a.glued())
        a.reverse()
        a.opposite()
        a.next()
        a.opposite()
    for corner in edge.Corners:
        self.delete_tet(corner.Tetrahedron)
    self.build_edge_classes()
    self.orient()
    return True