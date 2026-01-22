from sympy.core.numbers import (AlgebraicNumber, E, Float, I, Integer,
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.polys.polytools import Poly
from sympy.abc import x, y, z
from sympy.external.gmpy import HAS_GMPY
from sympy.polys.domains import (ZZ, QQ, RR, CC, FF, GF, EX, EXRAW, ZZ_gmpy,
from sympy.polys.domains.algebraicfield import AlgebraicField
from sympy.polys.domains.gaussiandomains import ZZ_I, QQ_I
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.domains.realfield import RealField
from sympy.polys.numberfields.subfield import field_isomorphism
from sympy.polys.rings import ring
from sympy.polys.specialpolys import cyclotomic_poly
from sympy.polys.fields import field
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.polyerrors import (
from sympy.testing.pytest import raises
from itertools import product
def test_ModularInteger():
    F3 = FF(3)
    a = F3(0)
    assert isinstance(a, F3.dtype) and a == 0
    a = F3(1)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2)
    assert isinstance(a, F3.dtype) and a == 2
    a = F3(3)
    assert isinstance(a, F3.dtype) and a == 0
    a = F3(4)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(F3(0))
    assert isinstance(a, F3.dtype) and a == 0
    a = F3(F3(1))
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(F3(2))
    assert isinstance(a, F3.dtype) and a == 2
    a = F3(F3(3))
    assert isinstance(a, F3.dtype) and a == 0
    a = F3(F3(4))
    assert isinstance(a, F3.dtype) and a == 1
    a = -F3(1)
    assert isinstance(a, F3.dtype) and a == 2
    a = -F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = 2 + F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) + 2
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) + F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) + F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = 3 - F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(3) - 2
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(3) - F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(3) - F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = 2 * F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) * 2
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) * F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) * F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = 2 / F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) / 2
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) / F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) / F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = 1 % F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(1) % 2
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(1) % F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(1) % F3(2)
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) ** 0
    assert isinstance(a, F3.dtype) and a == 1
    a = F3(2) ** 1
    assert isinstance(a, F3.dtype) and a == 2
    a = F3(2) ** 2
    assert isinstance(a, F3.dtype) and a == 1
    F7 = FF(7)
    a = F7(3) ** 100000000000
    assert isinstance(a, F7.dtype) and a == 4
    a = F7(3) ** (-100000000000)
    assert isinstance(a, F7.dtype) and a == 2
    a = F7(3) ** S(2)
    assert isinstance(a, F7.dtype) and a == 2
    assert bool(F3(3)) is False
    assert bool(F3(4)) is True
    F5 = FF(5)
    a = F5(1) ** (-1)
    assert isinstance(a, F5.dtype) and a == 1
    a = F5(2) ** (-1)
    assert isinstance(a, F5.dtype) and a == 3
    a = F5(3) ** (-1)
    assert isinstance(a, F5.dtype) and a == 2
    a = F5(4) ** (-1)
    assert isinstance(a, F5.dtype) and a == 4
    assert (F5(1) < F5(2)) is True
    assert (F5(1) <= F5(2)) is True
    assert (F5(1) > F5(2)) is False
    assert (F5(1) >= F5(2)) is False
    assert (F5(3) < F5(2)) is False
    assert (F5(3) <= F5(2)) is False
    assert (F5(3) > F5(2)) is True
    assert (F5(3) >= F5(2)) is True
    assert (F5(1) < F5(7)) is True
    assert (F5(1) <= F5(7)) is True
    assert (F5(1) > F5(7)) is False
    assert (F5(1) >= F5(7)) is False
    assert (F5(3) < F5(7)) is False
    assert (F5(3) <= F5(7)) is False
    assert (F5(3) > F5(7)) is True
    assert (F5(3) >= F5(7)) is True
    assert (F5(1) < 2) is True
    assert (F5(1) <= 2) is True
    assert (F5(1) > 2) is False
    assert (F5(1) >= 2) is False
    assert (F5(3) < 2) is False
    assert (F5(3) <= 2) is False
    assert (F5(3) > 2) is True
    assert (F5(3) >= 2) is True
    assert (F5(1) < 7) is True
    assert (F5(1) <= 7) is True
    assert (F5(1) > 7) is False
    assert (F5(1) >= 7) is False
    assert (F5(3) < 7) is False
    assert (F5(3) <= 7) is False
    assert (F5(3) > 7) is True
    assert (F5(3) >= 7) is True
    raises(NotInvertible, lambda: F5(0) ** (-1))
    raises(NotInvertible, lambda: F5(5) ** (-1))
    raises(ValueError, lambda: FF(0))
    raises(ValueError, lambda: FF(2.1))