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
def test_quaternion_conversions():
    q1 = Quaternion(1, 2, 3, 4)
    assert q1.to_axis_angle() == ((2 * sqrt(29) / 29, 3 * sqrt(29) / 29, 4 * sqrt(29) / 29), 2 * acos(sqrt(30) / 30))
    assert q1.to_rotation_matrix() == Matrix([[Rational(-2, 3), Rational(2, 15), Rational(11, 15)], [Rational(2, 3), Rational(-1, 3), Rational(2, 3)], [Rational(1, 3), Rational(14, 15), Rational(2, 15)]])
    assert q1.to_rotation_matrix((1, 1, 1)) == Matrix([[Rational(-2, 3), Rational(2, 15), Rational(11, 15), Rational(4, 5)], [Rational(2, 3), Rational(-1, 3), Rational(2, 3), S.Zero], [Rational(1, 3), Rational(14, 15), Rational(2, 15), Rational(-2, 5)], [S.Zero, S.Zero, S.Zero, S.One]])
    theta = symbols('theta', real=True)
    q2 = Quaternion(cos(theta / 2), 0, 0, sin(theta / 2))
    assert trigsimp(q2.to_rotation_matrix()) == Matrix([[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]])
    assert q2.to_axis_angle() == ((0, 0, sin(theta / 2) / Abs(sin(theta / 2))), 2 * acos(cos(theta / 2)))
    assert trigsimp(q2.to_rotation_matrix((1, 1, 1))) == Matrix([[cos(theta), -sin(theta), 0, sin(theta) - cos(theta) + 1], [sin(theta), cos(theta), 0, -sin(theta) - cos(theta) + 1], [0, 0, 1, 0], [0, 0, 0, 1]])