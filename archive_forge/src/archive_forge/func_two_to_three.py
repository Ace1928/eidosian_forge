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
def two_to_three(self, face_or_arrow, tet=None, return_arrow=False, must_succeed=False, unsafe_mode=False):
    """
        Perform a 2-to-3 Pachner move on the face specified by
        (face_or_arrow, tet), replacing the two tetrahedra with three
        tetrahedra around an edge.

        Returns ``True`` or ``False`` depending on whether the
        requested move succeeded.  When ``must_succeed`` is ``True``,
        it instead raises an exception if the requested move is
        topologically impossible.

        When ``unsafe_mode`` is ``True`` it does not rebuild the edge
        classes; in any mode, it does not rebuild the vertex classes.
        """
    if isinstance(face_or_arrow, Arrow):
        assert tet is None
        arrow = face_or_arrow
        a = arrow.copy()
    else:
        arrow = None
        a = Arrow(PickAnEdge[face_or_arrow], face_or_arrow, tet)
    a = a.copy()
    b = a.glued()
    if not unsafe_mode:
        possible, reason = self._face_permits_two_to_three(a, b)
        if not possible:
            if must_succeed:
                raise ValueError(reason)
            return False
    a_orig = a.copy()
    new = self.new_arrows(3)
    for i in range(3):
        new[i].glue(new[(i + 1) % 3])
    a.reverse()
    for c in new:
        c.opposite().glue(a.glued())
        c.reverse().glue(b.glued())
        a.rotate(-1)
        b.rotate(1)
    for c in new:
        c.reverse()
        c.opposite()
    self._two_to_three_move_hook(a_orig, new)
    self.delete_tet(a.Tetrahedron)
    self.delete_tet(b.Tetrahedron)
    if not unsafe_mode:
        self.build_edge_classes()
    if VERBOSE:
        print('2->3')
        print(self.EdgeValences)
    if return_arrow:
        return new[1].north_head().get_arrow()
    else:
        return True