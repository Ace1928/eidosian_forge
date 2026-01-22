from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import (I, Rational, pi)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.abc import alpha, beta, gamma, j, m
from sympy.physics.quantum import hbar, represent, Commutator, InnerProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.cg import CG
from sympy.physics.quantum.spin import (
from sympy.testing.pytest import raises, slow
def test_represent_spin_states():
    assert represent(JxKet(S.Half, S.Half), basis=Jx) == Matrix([1, 0])
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jx) == Matrix([0, 1])
    assert represent(JxKet(1, 1), basis=Jx) == Matrix([1, 0, 0])
    assert represent(JxKet(1, 0), basis=Jx) == Matrix([0, 1, 0])
    assert represent(JxKet(1, -1), basis=Jx) == Matrix([0, 0, 1])
    assert represent(JyKet(S.Half, S.Half), basis=Jx) == Matrix([exp(-I * pi / 4), 0])
    assert represent(JyKet(S.Half, Rational(-1, 2)), basis=Jx) == Matrix([0, exp(I * pi / 4)])
    assert represent(JyKet(1, 1), basis=Jx) == Matrix([-I, 0, 0])
    assert represent(JyKet(1, 0), basis=Jx) == Matrix([0, 1, 0])
    assert represent(JyKet(1, -1), basis=Jx) == Matrix([0, 0, I])
    assert represent(JzKet(S.Half, S.Half), basis=Jx) == sqrt(2) * Matrix([-1, 1]) / 2
    assert represent(JzKet(S.Half, Rational(-1, 2)), basis=Jx) == sqrt(2) * Matrix([-1, -1]) / 2
    assert represent(JzKet(1, 1), basis=Jx) == Matrix([1, -sqrt(2), 1]) / 2
    assert represent(JzKet(1, 0), basis=Jx) == sqrt(2) * Matrix([1, 0, -1]) / 2
    assert represent(JzKet(1, -1), basis=Jx) == Matrix([1, sqrt(2), 1]) / 2
    assert represent(JxKet(S.Half, S.Half), basis=Jy) == Matrix([exp(I * pi * Rational(-3, 4)), 0])
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jy) == Matrix([0, exp(I * pi * Rational(3, 4))])
    assert represent(JxKet(1, 1), basis=Jy) == Matrix([I, 0, 0])
    assert represent(JxKet(1, 0), basis=Jy) == Matrix([0, 1, 0])
    assert represent(JxKet(1, -1), basis=Jy) == Matrix([0, 0, -I])
    assert represent(JyKet(S.Half, S.Half), basis=Jy) == Matrix([1, 0])
    assert represent(JyKet(S.Half, Rational(-1, 2)), basis=Jy) == Matrix([0, 1])
    assert represent(JyKet(1, 1), basis=Jy) == Matrix([1, 0, 0])
    assert represent(JyKet(1, 0), basis=Jy) == Matrix([0, 1, 0])
    assert represent(JyKet(1, -1), basis=Jy) == Matrix([0, 0, 1])
    assert represent(JzKet(S.Half, S.Half), basis=Jy) == sqrt(2) * Matrix([-1, I]) / 2
    assert represent(JzKet(S.Half, Rational(-1, 2)), basis=Jy) == sqrt(2) * Matrix([I, -1]) / 2
    assert represent(JzKet(1, 1), basis=Jy) == Matrix([1, -I * sqrt(2), -1]) / 2
    assert represent(JzKet(1, 0), basis=Jy) == Matrix([-sqrt(2) * I, 0, -sqrt(2) * I]) / 2
    assert represent(JzKet(1, -1), basis=Jy) == Matrix([-1, -sqrt(2) * I, 1]) / 2
    assert represent(JxKet(S.Half, S.Half), basis=Jz) == sqrt(2) * Matrix([1, 1]) / 2
    assert represent(JxKet(S.Half, Rational(-1, 2)), basis=Jz) == sqrt(2) * Matrix([-1, 1]) / 2
    assert represent(JxKet(1, 1), basis=Jz) == Matrix([1, sqrt(2), 1]) / 2
    assert represent(JxKet(1, 0), basis=Jz) == sqrt(2) * Matrix([-1, 0, 1]) / 2
    assert represent(JxKet(1, -1), basis=Jz) == Matrix([1, -sqrt(2), 1]) / 2
    assert represent(JyKet(S.Half, S.Half), basis=Jz) == sqrt(2) * Matrix([-1, -I]) / 2
    assert represent(JyKet(S.Half, Rational(-1, 2)), basis=Jz) == sqrt(2) * Matrix([-I, -1]) / 2
    assert represent(JyKet(1, 1), basis=Jz) == Matrix([1, sqrt(2) * I, -1]) / 2
    assert represent(JyKet(1, 0), basis=Jz) == sqrt(2) * Matrix([I, 0, I]) / 2
    assert represent(JyKet(1, -1), basis=Jz) == Matrix([-1, sqrt(2) * I, 1]) / 2
    assert represent(JzKet(S.Half, S.Half), basis=Jz) == Matrix([1, 0])
    assert represent(JzKet(S.Half, Rational(-1, 2)), basis=Jz) == Matrix([0, 1])
    assert represent(JzKet(1, 1), basis=Jz) == Matrix([1, 0, 0])
    assert represent(JzKet(1, 0), basis=Jz) == Matrix([0, 1, 0])
    assert represent(JzKet(1, -1), basis=Jz) == Matrix([0, 0, 1])