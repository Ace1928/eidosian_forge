from . import constants
from . import exceptions
from . import epsilons
from .line import distance_r13_lines, R13Line, R13LineWithMatrix
from .geodesic_info import GeodesicInfo, LiftedTetrahedron
from .quotient_space import balance_end_points_of_line, ZQuotientLiftedTetrahedronSet
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..matrix import matrix # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
import heapq
from typing import Sequence, Any
def triangle_bounding_plane(tet, face, edge):
    v = tet.R13_vertices[face - edge]
    v0 = tet.R13_vertices[simplex.Head[edge]]
    v1 = tet.R13_vertices[simplex.Tail[edge]]
    m = time_r13_normalise(v0 / -r13_dot(v0, v) + v1 / -r13_dot(v1, v))
    return make_r13_unit_tangent_vector(m - v, m)