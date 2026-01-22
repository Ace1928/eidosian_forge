from .geodesic_info import GeodesicInfo
from .line import R13LineWithMatrix, distance_r13_lines
from . import constants
from . import epsilons
from . import exceptions
from ..snap.t3mlite import simplex, Tetrahedron, Mcomplex # type: ignore
from ..hyperboloid import r13_dot # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, List
def compute_plane_intersection_param(plane, point, direction, verified: bool):
    """
    Compute for which p the ray point + p * direction intersects the
    given plane, that is r13_dot(plane, point + p * direction) = 0.

    Note that when verified is true and intervals are given, only the
    positive possible values will be returned. That is, if the direction
    lies in the plane or is close to lying in the plane, the possible
    values are of the form (-inf, a) and (b, inf). In this case, the function
    returns the interval (b, inf) rather than (-inf, inf).
    """
    num = -r13_dot(plane, point)
    denom = r13_dot(plane, direction)
    if verified:
        if not denom != 0:
            return abs(num) / abs(denom)
    elif denom == 0:
        RF = denom.parent()
        denom = RF(1e-200)
    return num / denom