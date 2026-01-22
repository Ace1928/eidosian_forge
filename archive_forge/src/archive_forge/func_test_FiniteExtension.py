from sympy.core.symbol import symbols
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.polys import QQ, ZZ
from sympy.polys.polytools import Poly
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.agca.extensions import FiniteExtension
from sympy.polys.domainmatrix import DomainMatrix
from sympy.testing.pytest import raises
from sympy.abc import x, y, t
def test_FiniteExtension():
    A = FiniteExtension(Poly(x ** 2 + 1, x))
    assert A.rank == 2
    assert str(A) == 'ZZ[x]/(x**2 + 1)'
    i = A.generator
    assert i.parent() is A
    assert i * i == A(-1)
    raises(TypeError, lambda: i * ())
    assert A.basis == (A.one, i)
    assert A(1) == A.one
    assert i ** 2 == A(-1)
    assert i ** 2 != -1
    assert (2 + i) * (1 - i) == 3 - i
    assert (1 + i) ** 8 == A(16)
    assert A(1).inverse() == A(1)
    raises(NotImplementedError, lambda: A(2).inverse())
    F = FiniteExtension(Poly(x ** 3 - x + 1, x, modulus=3))
    assert F.rank == 3
    a = F.generator
    assert F.basis == (F(1), a, a ** 2)
    assert a ** 27 == a
    assert a ** 26 == F(1)
    assert a ** 13 == F(-1)
    assert a ** 9 == a + 1
    assert a ** 3 == a - 1
    assert a ** 6 == a ** 2 + a + 1
    assert F(x ** 2 + x).inverse() == 1 - a
    assert F(x + 2) ** (-1) == F(x + 2).inverse()
    assert a ** 19 * a ** (-19) == F(1)
    assert (a - 1) / (2 * a ** 2 - 1) == a ** 2 + 1
    assert (a - 1) // (2 * a ** 2 - 1) == a ** 2 + 1
    assert 2 / (a ** 2 + 1) == a ** 2 - a + 1
    assert (a ** 2 + 1) / 2 == -a ** 2 - 1
    raises(NotInvertible, lambda: F(0).inverse())
    K = FiniteExtension(Poly(t ** 2 - x ** 3 - x + 1, t, field=True))
    assert K.rank == 2
    assert str(K) == 'ZZ(x)[t]/(t**2 - x**3 - x + 1)'
    y = K.generator
    c = 1 / (x ** 3 - x ** 2 + x - 1)
    assert ((y + x) * (y - x)).inverse() == K(c)
    assert (y + x) * (y - x) * c == K(1)