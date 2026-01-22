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
def test_represent_coupled_states():
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == Matrix([1, 0, 0, 0])
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == Matrix([0, 1, 0, 0])
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == Matrix([0, 0, 1, 0])
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == Matrix([0, 0, 0, 1])
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == Matrix([1, 0, 0, 0])
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == Matrix([0, -I, 0, 0])
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == Matrix([0, 0, 1, 0])
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == Matrix([0, 0, 0, I])
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jx) == Matrix([1, 0, 0, 0])
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jx) == Matrix([0, S.Half, -sqrt(2) / 2, S.Half])
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jx) == Matrix([0, sqrt(2) / 2, 0, -sqrt(2) / 2])
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jx) == Matrix([0, S.Half, sqrt(2) / 2, S.Half])
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == Matrix([1, 0, 0, 0])
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == Matrix([0, I, 0, 0])
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == Matrix([0, 0, 1, 0])
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == Matrix([0, 0, 0, -I])
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == Matrix([1, 0, 0, 0])
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == Matrix([0, 1, 0, 0])
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == Matrix([0, 0, 1, 0])
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == Matrix([0, 0, 0, 1])
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jy) == Matrix([1, 0, 0, 0])
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jy) == Matrix([0, S.Half, -I * sqrt(2) / 2, Rational(-1, 2)])
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jy) == Matrix([0, -I * sqrt(2) / 2, 0, -I * sqrt(2) / 2])
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jy) == Matrix([0, Rational(-1, 2), -I * sqrt(2) / 2, S.Half])
    assert represent(JxKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == Matrix([1, 0, 0, 0])
    assert represent(JxKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == Matrix([0, S.Half, sqrt(2) / 2, S.Half])
    assert represent(JxKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == Matrix([0, -sqrt(2) / 2, 0, sqrt(2) / 2])
    assert represent(JxKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == Matrix([0, S.Half, -sqrt(2) / 2, S.Half])
    assert represent(JyKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == Matrix([1, 0, 0, 0])
    assert represent(JyKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == Matrix([0, S.Half, I * sqrt(2) / 2, Rational(-1, 2)])
    assert represent(JyKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == Matrix([0, I * sqrt(2) / 2, 0, I * sqrt(2) / 2])
    assert represent(JyKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == Matrix([0, Rational(-1, 2), I * sqrt(2) / 2, S.Half])
    assert represent(JzKetCoupled(0, 0, (S.Half, S.Half)), basis=Jz) == Matrix([1, 0, 0, 0])
    assert represent(JzKetCoupled(1, 1, (S.Half, S.Half)), basis=Jz) == Matrix([0, 1, 0, 0])
    assert represent(JzKetCoupled(1, 0, (S.Half, S.Half)), basis=Jz) == Matrix([0, 0, 1, 0])
    assert represent(JzKetCoupled(1, -1, (S.Half, S.Half)), basis=Jz) == Matrix([0, 0, 0, 1])