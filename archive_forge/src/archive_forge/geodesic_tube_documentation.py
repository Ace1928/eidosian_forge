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

        Finds the pending piece "closest" to the lifted closed geodesic,
        adds it to the result and marks the neighboring lifted tetrahedra
        to the pending queue.

        Here, "closest" is not quite precise because we pick the piece
        with the lowest lower bound for the distance. Also recall that the
        distance of a pending piece is the distance between the lifted
        geodesic L and the entry cell of the lifted tetrahedron, not between
        L and the lifted tetrahedron itself.

        So the right picture to have in mind is: imagine the 2-skeleton
        of the triangulation in the quotient space intersecting the boundary
        of a geodesic tube. As the geodesic tube grows, the intersection
        sweeps through the 2-skeleton. The pending pieces will be processed in
        the order the faces of the 2-skeleton are encountered during the
        sweep.
        