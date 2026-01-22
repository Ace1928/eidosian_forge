from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def homology_of_cyclic_branched_cover(knot_exterior, p):
    C = knot_exterior.covers(p, cover_type='cyclic')[0]
    return [d for d in C.homology().elementary_divisors() if d != 0]