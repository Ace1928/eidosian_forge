from math import prod
from sympy import QQ, ZZ
from sympy.abc import x, theta
from sympy.ntheory import factorint
from sympy.ntheory.residue_ntheory import n_order
from sympy.polys import Poly, cyclotomic_poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.numberfields.basis import round_two
from sympy.polys.numberfields.exceptions import StructureError
from sympy.polys.numberfields.modules import PowerBasis, to_col
from sympy.polys.numberfields.primes import (
from sympy.testing.pytest import raises
def test_two_elt_rep():
    ell = 7
    T = Poly(cyclotomic_poly(ell))
    ZK, dK = round_two(T)
    for p in [29, 13, 11, 5]:
        P = prime_decomp(p, T)
        for Pi in P:
            H = p * ZK + Pi.alpha * ZK
            gens = H.basis_element_pullbacks()
            b = _two_elt_rep(gens, ZK, p)
            if b != Pi.alpha:
                H2 = p * ZK + b * ZK
                assert H2 == H