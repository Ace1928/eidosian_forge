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
def perturb_geodesic(geodesic: GeodesicInfo, injectivity_radius, verified: bool):
    if geodesic.line is None:
        raise ValueError('GeodesicInfo needs line to be perturbed.')
    perturbed_point = perturb_unit_time_point(time_r13_normalise(geodesic.unnormalised_start_point), max_amt=injectivity_radius, verified=verified)
    m = geodesic.line.o13_matrix
    geodesic.unnormalised_start_point = perturbed_point
    geodesic.unnormalised_end_point = m * perturbed_point
    geodesic.line = None
    geodesic.find_tet_or_core_curve()