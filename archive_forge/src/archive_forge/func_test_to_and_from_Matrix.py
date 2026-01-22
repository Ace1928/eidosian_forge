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
def test_to_and_from_Matrix():
    q = Quaternion(w, x, y, z)
    q_full = Quaternion.from_Matrix(q.to_Matrix())
    q_vect = Quaternion.from_Matrix(q.to_Matrix(True))
    assert (q - q_full).is_zero_quaternion()
    assert (q.vector_part() - q_vect).is_zero_quaternion()