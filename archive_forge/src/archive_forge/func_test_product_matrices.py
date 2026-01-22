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
def test_product_matrices():
    q1 = Quaternion(w, x, y, z)
    q2 = Quaternion(*symbols('a:d'))
    assert (q1 * q2).to_Matrix() == q1.product_matrix_left * q2.to_Matrix()
    assert (q1 * q2).to_Matrix() == q2.product_matrix_right * q1.to_Matrix()
    R1 = (q1.product_matrix_left * q1.product_matrix_right.T)[1:, 1:]
    R2 = simplify(q1.to_rotation_matrix() * q1.norm() ** 2)
    assert R1 == R2