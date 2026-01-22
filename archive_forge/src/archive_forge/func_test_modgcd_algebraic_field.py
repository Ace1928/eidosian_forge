from sympy.polys.rings import ring
from sympy.polys.domains import ZZ, QQ, AlgebraicField
from sympy.polys.modulargcd import (
from sympy.functions.elementary.miscellaneous import sqrt
def test_modgcd_algebraic_field():
    A = AlgebraicField(QQ, sqrt(2))
    R, x = ring('x', A)
    one = A.one
    f, g = (2 * x, R(2))
    assert func_field_modgcd(f, g) == (one, f, g)
    f, g = (2 * x, R(sqrt(2)))
    assert func_field_modgcd(f, g) == (one, f, g)
    f, g = (2 * x + 2, 6 * x ** 2 - 6)
    assert func_field_modgcd(f, g) == (x + 1, R(2), 6 * x - 6)
    R, x, y = ring('x, y', A)
    f, g = (x + sqrt(2) * y, x + y)
    assert func_field_modgcd(f, g) == (one, f, g)
    f, g = (x * y + sqrt(2) * y ** 2, R(sqrt(2)) * y)
    assert func_field_modgcd(f, g) == (y, x + sqrt(2) * y, R(sqrt(2)))
    f, g = (x ** 2 + 2 * sqrt(2) * x * y + 2 * y ** 2, x + sqrt(2) * y)
    assert func_field_modgcd(f, g) == (g, g, one)
    A = AlgebraicField(QQ, sqrt(2), sqrt(3))
    R, x, y, z = ring('x, y, z', A)
    h = x ** 2 * y ** 7 + sqrt(6) / 21 * z
    f, g = (h * (27 * y ** 3 + 1), h * (y + x))
    assert func_field_modgcd(f, g) == (h, 27 * y ** 3 + 1, y + x)
    h = x ** 13 * y ** 3 + 1 / 2 * x ** 10 + 1 / sqrt(2)
    f, g = (h * (x + 1), h * sqrt(2) / sqrt(3))
    assert func_field_modgcd(f, g) == (h, x + 1, R(sqrt(2) / sqrt(3)))
    A = AlgebraicField(QQ, sqrt(2) ** (-1) * sqrt(3))
    R, x = ring('x', A)
    f, g = (x + 1, x - 1)
    assert func_field_modgcd(f, g) == (A.one, f, g)