from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def nonzero_divisor_product(knot_exterior, p):
    """
       sage: M = Manifold('K12n731')
       sage: nonzero_divisor_product(M, 3)
       2704
    """
    C = knot_exterior.covers(p, cover_type='cyclic')[0]
    divisors = C.homology().elementary_divisors()
    ans = 1
    for d in divisors:
        if d != 0:
            ans *= d
    return ans