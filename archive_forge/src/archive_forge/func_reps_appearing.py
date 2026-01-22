from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def reps_appearing(knot_exterior, p, q):
    """
    All irreducible C_p reps appearing in the F_q homology of the
    cyclic branched cover B_p, together with their multiplicities.

       sage: M = Manifold('K12a169')
       sage: [(A.trace(), e) for A, e in reps_appearing(M, 3, 5)]
       [(4, 1)]
    """
    M = knot_exterior
    G = M.fundamental_group()
    for A in irreps(p, q)[1:]:
        d = dim_twisted_homology(G, A)
        if d > 0:
            n = A.nrows()
            assert d % n == 0
            yield (A, d // n)