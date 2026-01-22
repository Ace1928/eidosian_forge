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
def split_star(self, edge):
    """
        Subdivides the star of an edge e.  If the edge has an embedded
        star then this operation first subdivides the edge, producing
        one new vertex and two new edges.  Next each tetrahedron which
        meets the edge is divided into two tetrahedra along a face
        which is the join of the new vertex to the edge opposite to e.
        The edge e must not be self-adjacent in any 2-simplex for this
        operation to be possible.  However, it is allowed for a
        tetrahedron to have two opposite edges identified to e.  In
        this case the tetrahedron is split into four tetrahedra,
        forming the join of two segments of length 2.  In order to
        deal with this situation we work our way around the edge
        making the identifications as we go.  The first time that we
        encounter a corner of a certain tetrahedron it gets split into
        two.  Those two are glued into place and may be encountered
        later in the process, at which time each of them get split in
        two.

        Returns an arrow associated to the "top half" of the original edge
        and the "first" tetrahedron adjacent to that edge, or 0 if the edge
        is self-adjacent.
        """
    if edge.selfadjacent():
        return 0
    garbage = []
    first_arrow = edge.get_arrow().next()
    first_bottom, first_top = self.new_arrows(2)
    a = first_arrow.copy()
    bottom = first_bottom.copy()
    top = first_top.copy()
    while 1:
        garbage.append(a.Tetrahedron)
        bottom.glue(top)
        a.opposite()
        above = a.glued()
        if above.is_null():
            check = a.copy().opposite().reverse()
            new_first = top.copy().opposite().reverse()
            if check == first_top:
                first_top = new_first
            elif check == first_bottom:
                first_bottom = new_first
        else:
            top.glue(above)
        bottom.reverse()
        a.reverse()
        below = a.glued()
        if below.is_null():
            check = a.copy().opposite().reverse()
            new_first = bottom.copy().opposite().reverse()
            if check == first_bottom:
                first_bottom = new_first
            elif check == first_top:
                first_top = new_first
        else:
            bottom.glue(below)
        bottom.reverse()
        a.reverse()
        a.opposite()
        a.next()
        if a == first_arrow:
            break
        next_bottom, next_top = self.new_arrows(2)
        top.opposite()
        bottom.opposite()
        top.glue(next_top)
        bottom.glue(next_bottom)
        top = next_top.opposite()
        bottom = next_bottom.opposite()
    top.opposite()
    bottom.opposite()
    top.glue(first_top.opposite())
    bottom.glue(first_bottom.opposite())
    for tet in garbage:
        self.delete_tet(tet)
    self.rebuild()
    return first_top