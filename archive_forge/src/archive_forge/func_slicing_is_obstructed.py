from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def slicing_is_obstructed(knot_exterior, p, q):
    """
    Applies the test of Section 8 of [HKL] to the F_q homology of the
    branched cover B_p::

       sage: M = Manifold('K12n813')
       sage: slicing_is_obstructed(M, 2, 3)
       False
       sage: slicing_is_obstructed(M, 3, 7)
       True
    """
    reps = list(reps_appearing(knot_exterior, p, q))
    if len(reps) == 0:
        return False
    for A, e in reps:
        if e > 1:
            return False
        chi = lambda v: v[0]
        f = alex_poly_of_induced_rep(p, knot_exterior, A, chi)
        if poly_is_a_norm(f):
            return False
    return True