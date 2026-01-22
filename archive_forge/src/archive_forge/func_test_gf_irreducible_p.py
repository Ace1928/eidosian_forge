from sympy.polys.galoistools import (
from sympy.polys.polyerrors import (
from sympy.polys import polyconfig as config
from sympy.polys.domains import ZZ
from sympy.core.numbers import pi
from sympy.ntheory.generate import nextprime
from sympy.testing.pytest import raises
def test_gf_irreducible_p():
    assert gf_irred_p_ben_or(ZZ.map([7]), 11, ZZ) is True
    assert gf_irred_p_ben_or(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irred_p_ben_or(ZZ.map([7, 3, 1]), 11, ZZ) is False
    assert gf_irred_p_rabin(ZZ.map([7]), 11, ZZ) is True
    assert gf_irred_p_rabin(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irred_p_rabin(ZZ.map([7, 3, 1]), 11, ZZ) is False
    config.setup('GF_IRRED_METHOD', 'ben-or')
    assert gf_irreducible_p(ZZ.map([7]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3, 1]), 11, ZZ) is False
    config.setup('GF_IRRED_METHOD', 'rabin')
    assert gf_irreducible_p(ZZ.map([7]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3]), 11, ZZ) is True
    assert gf_irreducible_p(ZZ.map([7, 3, 1]), 11, ZZ) is False
    config.setup('GF_IRRED_METHOD', 'other')
    raises(KeyError, lambda: gf_irreducible_p([7], 11, ZZ))
    config.setup('GF_IRRED_METHOD')
    f = ZZ.map([1, 9, 9, 13, 16, 15, 6, 7, 7, 7, 10])
    g = ZZ.map([1, 7, 16, 7, 15, 13, 13, 11, 16, 10, 9])
    h = gf_mul(f, g, 17, ZZ)
    assert gf_irred_p_ben_or(f, 17, ZZ) is True
    assert gf_irred_p_ben_or(g, 17, ZZ) is True
    assert gf_irred_p_ben_or(h, 17, ZZ) is False
    assert gf_irred_p_rabin(f, 17, ZZ) is True
    assert gf_irred_p_rabin(g, 17, ZZ) is True
    assert gf_irred_p_rabin(h, 17, ZZ) is False