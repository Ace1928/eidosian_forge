from .fixed_points import r13_fixed_points_of_psl2c_matrix # type: ignore
from ..hyperboloid import r13_dot, o13_inverse # type: ignore
from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..sage_helper import _within_sage # type: ignore
def distance_r13_lines(line0: R13Line, line1: R13Line):
    """
    Computes distance between two hyperbolic lines.
    """
    p00 = r13_dot(line0.points[0], line1.points[0])
    p01 = r13_dot(line0.points[0], line1.points[1])
    p10 = r13_dot(line0.points[1], line1.points[0])
    p11 = r13_dot(line0.points[1], line1.points[1])
    pp = line0.inner_product * line1.inner_product
    t0 = _safe_sqrt(p00 * p11 / pp)
    t1 = _safe_sqrt(p01 * p10 / pp)
    p = (t0 + t1 - 1) / 2
    return 2 * _safe_sqrt(p).arcsinh()