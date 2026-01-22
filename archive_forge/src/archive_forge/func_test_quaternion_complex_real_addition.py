from sympy.core.function import diff
from sympy.core.function import expand
from sympy.core.numbers import (E, I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.functions.elementary.complexes import (Abs, conjugate, im, re, sign)
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (acos, asin, cos, sin, atan2, atan)
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import Matrix
from sympy.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.algebras.quaternion import Quaternion
from sympy.testing.pytest import raises
from itertools import permutations, product
def test_quaternion_complex_real_addition():
    a = symbols('a', complex=True)
    b = symbols('b', real=True)
    c = symbols('c', commutative=False)
    q = Quaternion(w, x, y, z)
    assert a + q == Quaternion(w + re(a), x + im(a), y, z)
    assert 1 + q == Quaternion(1 + w, x, y, z)
    assert I + q == Quaternion(w, 1 + x, y, z)
    assert b + q == Quaternion(w + b, x, y, z)
    raises(ValueError, lambda: c + q)
    raises(ValueError, lambda: q * c)
    raises(ValueError, lambda: c * q)
    assert -q == Quaternion(-w, -x, -y, -z)
    q1 = Quaternion(3 + 4 * I, 2 + 5 * I, 0, 7 + 8 * I, real_field=False)
    q2 = Quaternion(1, 4, 7, 8)
    assert q1 + (2 + 3 * I) == Quaternion(5 + 7 * I, 2 + 5 * I, 0, 7 + 8 * I)
    assert q2 + (2 + 3 * I) == Quaternion(3, 7, 7, 8)
    assert q1 * (2 + 3 * I) == Quaternion((2 + 3 * I) * (3 + 4 * I), (2 + 3 * I) * (2 + 5 * I), 0, (2 + 3 * I) * (7 + 8 * I))
    assert q2 * (2 + 3 * I) == Quaternion(-10, 11, 38, -5)
    q1 = Quaternion(1, 2, 3, 4)
    q0 = Quaternion(0, 0, 0, 0)
    assert q1 + q0 == q1
    assert q1 - q0 == q1
    assert q1 - q1 == q0