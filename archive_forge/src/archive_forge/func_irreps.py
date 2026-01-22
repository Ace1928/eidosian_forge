from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def irreps(p, q):
    """
    Returns the irreducible representations of the cyclic group C_p
    over the field F_q, where p and q are distinct primes.

    Each representation is given by a matrix over F_q giving the
    action of the preferred generator of C_p.

       sage: [M.nrows() for M in irreps(3, 7)]
       [1, 1, 1]
       sage: [M.nrows() for M in irreps(7, 11)]
       [1, 3, 3]
       sage: sum(M.nrows() for M in irreps(157, 13))
       157
    """
    p, q = (ZZ(p), ZZ(q))
    assert p.is_prime() and q.is_prime() and (p != q)
    R = PolynomialRing(GF(q), 'x')
    x = R.gen()
    polys = [f for f, e in (x ** p - 1).factor()]
    polys.sort(key=lambda f: (f.degree(), -f.constant_coefficient()))
    reps = [poly_to_rep(f) for f in polys]
    assert all((A ** p == 1 for A in reps))
    assert reps[0] == 1
    return reps