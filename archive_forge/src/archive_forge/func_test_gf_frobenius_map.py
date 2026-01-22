from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_frobenius_map():
    f = ZZ.map([2, 0, 1, 0, 2, 2, 0, 2, 2, 2])
    g = ZZ.map([1, 1, 0, 2, 0, 1, 0, 2, 0, 1])
    p = 3
    b = gf_frobenius_monomial_base(g, p, ZZ)
    h = gf_frobenius_map(f, g, b, p, ZZ)
    h1 = gf_pow_mod(f, p, g, p, ZZ)
    assert h == h1