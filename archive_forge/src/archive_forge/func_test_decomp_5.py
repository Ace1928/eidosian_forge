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
def test_decomp_5():
    for d in [-7, -3]:
        T = Poly(x ** 2 - d)
        rad = {}
        ZK, dK = round_two(T, radicals=rad)
        p = 2
        P = prime_decomp(p, T, dK=dK, ZK=ZK, radical=rad.get(p))
        if d % 8 == 1:
            assert len(P) == 2
            assert all((P[i].e == 1 and P[i].f == 1 for i in range(2)))
            assert prod((Pi ** Pi.e for Pi in P)) == p * ZK
        else:
            assert d % 8 == 5
            assert len(P) == 1
            assert P[0].e == 1
            assert P[0].f == 2
            assert P[0].as_submodule() == p * ZK