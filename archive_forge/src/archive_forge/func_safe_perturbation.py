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
def safe_perturbation(self):
    """
        Return an integer N so that moving all the vertices in the PL
        link independently by at most 1/N (per coordinate) cannot change
        the topology.
        """
    min_distance_sq = 2 ** (-12)
    for tet in self:
        m = len(tet.arcs)
        points_3d = [arc.to_3d_points() for arc in tet.arcs]
        coor_min = [[min(x[i], y[i]) for i in range(3)] for x, y in points_3d]
        coor_max = [[max(x[i], y[i]) for i in range(3)] for x, y in points_3d]
        for a in range(m):
            arc_a = tet.arcs[a]
            min_a, max_a = (coor_min[a], coor_max[a])
            for b in range(a + 1, m):
                arc_b = tet.arcs[b]
                if arc_a != arc_b.past and arc_a != arc_b.next:
                    check = True
                    min_b, max_b = (coor_min[b], coor_max[b])
                    for i in range(3):
                        if (min_a[i] - max_b[i]) ** 2 >= min_distance_sq and min_a[i] - max_b[i] > 0 or ((min_b[i] - max_a[i]) ** 2 >= min_distance_sq and min_b[i] - max_a[i] > 0):
                            check = False
                            break
                    if check:
                        d2 = pl_utils.arc_distance_sq(points_3d[a], points_3d[b])
                        min_distance_sq = min(d2, min_distance_sq)
    return int(4 / pl_utils.rational_sqrt(min_distance_sq)) + 1