from sympy.polys.distributedmodules import (
from sympy.polys.orderings import lex, grlex, InverseOrder
from sympy.polys.domains import QQ
from sympy.abc import x, y, z
def test_nontrivial():
    gens = [x, y, z]

    def contains(I, f):
        S = [sdm_from_vector([g], lex, QQ, gens=gens) for g in I]
        G = sdm_groebner(S, sdm_nf_mora, lex, QQ)
        return sdm_nf_mora(sdm_from_vector([f], lex, QQ, gens=gens), G, lex, QQ) == sdm_zero()
    assert contains([x, y], x)
    assert contains([x, y], x + y)
    assert not contains([x, y], 1)
    assert not contains([x, y], z)
    assert contains([x ** 2 + y, x ** 2 + x], x - y)
    assert not contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 2)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 3)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 4)
    assert not contains([x + y + z, x * y + x * z + y * z, x * y * z], x * y ** 2)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x ** 4 + y ** 3 + 2 * z * y * x)
    assert contains([x + y + z, x * y + x * z + y * z, x * y * z], x * y * z)
    assert contains([x, 1 + x + y, 5 - 7 * y], 1)
    assert contains([x ** 3 + y ** 3, y ** 3 + z ** 3, z ** 3 + x ** 3, x ** 2 * y + x ** 2 * z + y ** 2 * z], x ** 3)
    assert not contains([x ** 3 + y ** 3, y ** 3 + z ** 3, z ** 3 + x ** 3, x ** 2 * y + x ** 2 * z + y ** 2 * z], x ** 2 + y ** 2)
    assert not contains([x * (1 + x + y), y * (1 + z)], x)
    assert not contains([x * (1 + x + y), y * (1 + z)], x + y)