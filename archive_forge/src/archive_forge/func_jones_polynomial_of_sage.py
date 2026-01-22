import spherogram
import snappy
from sage.all import Link as SageLink
from sage.all import LaurentPolynomialRing, PolynomialRing, ZZ, var
from sage.symbolic.ring import SymbolicRing
def jones_polynomial_of_sage(knot):
    """
    To match our conventions (which seem to agree with KnotAtlas), we
    need to swap q and 1/q.
    """
    q = LaurentPolynomialRing(ZZ, 'q').gen()
    p = knot.jones_polynomial(skein_normalization=True)
    exps = p.exponents()
    assert all((e % 4 == 0 for e in exps))
    return sum((c * q ** (-e // 4) for c, e in zip(p.coefficients(), exps)))