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
def check_arcs_are_embedded(self):
    for tet in manifold:
        for arc0, arc1 in itertools.combinations(tet.arcs, 2):
            a, b = arc0.to_3d_points()
            c, d = arc1.to_3d_points()
            assert not pl_utils.segments_meet_not_at_endpoint((a, b), (c, d))