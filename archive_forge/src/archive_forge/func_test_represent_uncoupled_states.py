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
def test_represent_uncoupled_states():
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jx) == Matrix([1, 0, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jx) == Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jx) == Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jx) == Matrix([0, 0, 0, 1])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jx) == Matrix([-I, 0, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jx) == Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jx) == Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jx) == Matrix([0, 0, 0, I])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jx) == Matrix([S.Half, Rational(-1, 2), Rational(-1, 2), S.Half])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jx) == Matrix([S.Half, S.Half, Rational(-1, 2), Rational(-1, 2)])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jx) == Matrix([S.Half, Rational(-1, 2), S.Half, Rational(-1, 2)])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jx) == Matrix([S.Half, S.Half, S.Half, S.Half])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jy) == Matrix([I, 0, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jy) == Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jy) == Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jy) == Matrix([0, 0, 0, -I])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jy) == Matrix([1, 0, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jy) == Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jy) == Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jy) == Matrix([0, 0, 0, 1])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jy) == Matrix([S.Half, -I / 2, -I / 2, Rational(-1, 2)])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jy) == Matrix([-I / 2, S.Half, Rational(-1, 2), -I / 2])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jy) == Matrix([-I / 2, Rational(-1, 2), S.Half, -I / 2])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jy) == Matrix([Rational(-1, 2), -I / 2, -I / 2, S.Half])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, S.Half)), basis=Jz) == Matrix([S.Half, S.Half, S.Half, S.Half])
    assert represent(TensorProduct(JxKet(S.Half, S.Half), JxKet(S.Half, Rational(-1, 2))), basis=Jz) == Matrix([Rational(-1, 2), S.Half, Rational(-1, 2), S.Half])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, S.Half)), basis=Jz) == Matrix([Rational(-1, 2), Rational(-1, 2), S.Half, S.Half])
    assert represent(TensorProduct(JxKet(S.Half, Rational(-1, 2)), JxKet(S.Half, Rational(-1, 2))), basis=Jz) == Matrix([S.Half, Rational(-1, 2), Rational(-1, 2), S.Half])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, S.Half)), basis=Jz) == Matrix([S.Half, I / 2, I / 2, Rational(-1, 2)])
    assert represent(TensorProduct(JyKet(S.Half, S.Half), JyKet(S.Half, Rational(-1, 2))), basis=Jz) == Matrix([I / 2, S.Half, Rational(-1, 2), I / 2])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, S.Half)), basis=Jz) == Matrix([I / 2, Rational(-1, 2), S.Half, I / 2])
    assert represent(TensorProduct(JyKet(S.Half, Rational(-1, 2)), JyKet(S.Half, Rational(-1, 2))), basis=Jz) == Matrix([Rational(-1, 2), I / 2, I / 2, S.Half])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, S.Half)), basis=Jz) == Matrix([1, 0, 0, 0])
    assert represent(TensorProduct(JzKet(S.Half, S.Half), JzKet(S.Half, Rational(-1, 2))), basis=Jz) == Matrix([0, 1, 0, 0])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, S.Half)), basis=Jz) == Matrix([0, 0, 1, 0])
    assert represent(TensorProduct(JzKet(S.Half, Rational(-1, 2)), JzKet(S.Half, Rational(-1, 2))), basis=Jz) == Matrix([0, 0, 0, 1])