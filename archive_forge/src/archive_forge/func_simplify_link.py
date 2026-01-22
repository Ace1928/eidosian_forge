from ..snap.t3mlite.simplex import *
from ..snap.t3mlite.edge import Edge
from ..snap.t3mlite.arrow import Arrow
from ..snap.t3mlite.tetrahedron import Tetrahedron
from ..snap.t3mlite.mcomplex import VERBOSE
from .exceptions import GeneralPositionError
from .rational_linear_algebra import Vector3, QQ
from . import pl_utils
from . import stored_moves
from .mcomplex_with_expansion import McomplexWithExpansion
from .mcomplex_with_memory import McomplexWithMemory
from .barycentric_geometry import (BarycentricPoint, BarycentricArc,
import random
import collections
import time
def simplify_link(self, tet, straighten=True, push=True):
    if straighten:
        any_success = straighten_arcs(tet.arcs)
    if push:
        success = True
        while success:
            success = False
            tri = pushable_tri_in_tet(tet.arcs)
            if tri is not None:
                if self.push_through_face(tri, tet):
                    success, any_success = (True, True)
    return any_success