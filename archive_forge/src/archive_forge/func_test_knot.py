import spherogram
import snappy
from sage.all import Link as SageLink
from sage.all import LaurentPolynomialRing, PolynomialRing, ZZ, var
from sage.symbolic.ring import SymbolicRing
def test_knot(snappy_manifold):
    M = snappy_manifold
    assert M.num_cusps() == 1
    U = M.link()
    T = U.sage_link()
    assert [list(x) for x in U.PD_code(min_strand_index=1)] == T.pd_code()
    assert U.alexander_polynomial() == alexander_poly_of_sage(T)
    assert U.signature() == T.signature()
    assert U.jones_polynomial() == jones_polynomial_of_sage(T)
    T_alt = SageLink(U.braid_word(as_sage_braid=True))
    assert U.signature() == T_alt.signature()
    U_alt = spherogram.Link(T.braid())
    assert U.signature() == U_alt.signature()