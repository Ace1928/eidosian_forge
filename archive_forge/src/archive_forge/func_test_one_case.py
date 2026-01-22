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
def test_one_case(q):
    angles1 = Matrix(q.to_euler(seq, True, True))
    angles2 = Matrix(q.to_euler(seq, False, False))
    angle_errors = simplify(angles1 - angles2).evalf()
    for angle_error in angle_errors:
        angle_error = (angle_error + pi) % (2 * pi) - pi
        assert angle_error < 1e-06