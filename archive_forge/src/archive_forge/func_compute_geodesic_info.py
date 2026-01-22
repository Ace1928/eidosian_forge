from . import exceptions
from . import epsilons
from . import debug
from .tracing import trace_geodesic
from .crush import crush_geodesic_pieces
from .line import R13LineWithMatrix
from .geometric_structure import add_r13_geometry, word_to_psl2c_matrix
from .geodesic_info import GeodesicInfo, sample_line
from .perturb import perturb_geodesics
from .subdivide import traverse_geodesics_to_subdivide
from .cusps import (
from ..snap.t3mlite import Mcomplex
from ..exceptions import InsufficientPrecisionError
import functools
from typing import Sequence
def compute_geodesic_info(mcomplex: Mcomplex, word) -> GeodesicInfo:
    """
    Compute basic information about a geodesic given a word.

    add_r13_geometry must have been called on the Mcomplex.
    """
    m = word_to_psl2c_matrix(mcomplex, word)
    _verify_not_parabolic(m, mcomplex, word)
    line = R13LineWithMatrix.from_psl2c_matrix(m)
    start_point = sample_line(line)
    g = GeodesicInfo(mcomplex=mcomplex, trace=m.trace(), unnormalised_start_point=start_point, unnormalised_end_point=line.o13_matrix * start_point, line=line)
    g.find_tet_or_core_curve()
    return g