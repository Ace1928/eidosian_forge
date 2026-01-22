from sympy.polys.groebnertools import (
from sympy.polys.fglmtools import _representing_matrices
from sympy.polys.orderings import lex, grlex
from sympy.polys.rings import ring, xring
from sympy.polys.domains import ZZ, QQ
from sympy.testing.pytest import slow
from sympy.polys import polyconfig as config
def test_sig_key():
    s1 = sig((0,) * 3, 2)
    s2 = sig((1,) * 3, 4)
    s3 = sig((2,) * 3, 2)
    assert sig_key(s1, lex) > sig_key(s2, lex)
    assert sig_key(s2, lex) < sig_key(s3, lex)