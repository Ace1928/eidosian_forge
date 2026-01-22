from . import constants
from . import epsilons
from . import exceptions
from .geodesic_tube import add_structures_necessary_for_tube, GeodesicTube
from .geodesic_info import GeodesicInfo
from .line import R13Line, distance_r13_lines
from ..hyperboloid import ( # type: ignore
from ..snap.t3mlite import Mcomplex # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from ..matrix import vector # type: ignore
from ..math_basics import correct_min # type: ignore
from typing import Sequence, List
def compute_lower_bound_injectivity_radius_from_tubes(mcomplex: Mcomplex, tubes: Sequence[GeodesicTube]):
    if len(tubes) == 0:
        raise Exception('No geodesic tubes given')
    distances = []
    tet_to_lines: List[List[R13Line]] = [[] for tet in mcomplex.Tetrahedra]
    for tube in tubes:
        distances.append(tube.covered_radius())
        for p in tube.pieces:
            tet_to_lines[p.tet.Index].append(p.lifted_geodesic)
    for tet in mcomplex.Tetrahedra:
        for curve in tet.core_curves.values():
            tet_to_lines[tet.Index].append(curve.r13_line)
    for r13_lines in tet_to_lines:
        for i, r13_line0 in enumerate(r13_lines):
            for r13_line1 in r13_lines[:i]:
                distances.append(distance_r13_lines(r13_line0, r13_line1))
    return correct_min(distances) / 2