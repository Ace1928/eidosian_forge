from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
from sympy.functions.elementary.miscellaneous import sqrt
def test_to_ZZ_ANP_poly():
    A = AlgebraicField(QQ, sqrt(2))
    R, x = ring('x', A)
    f = x * (sqrt(2) + 1)
    T, x_, z_ = ring('x_, z_', ZZ)
    f_ = x_ * z_ + x_
    assert _to_ZZ_poly(f, T) == f_
    assert _to_ANP_poly(f_, R) == f
    R, x, t, s = ring('x, t, s', A)
    f = x * t ** 2 + x * s + sqrt(2)
    D, t_, s_ = ring('t_, s_', ZZ)
    T, x_, z_ = ring('x_, z_', D)
    f_ = (t_ ** 2 + s_) * x_ + z_
    assert _to_ZZ_poly(f, T) == f_
    assert _to_ANP_poly(f_, R) == f