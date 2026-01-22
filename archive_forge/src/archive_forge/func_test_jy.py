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
def test_jy():
    assert Commutator(Jy, Jz).doit() == I * hbar * Jx
    assert Jy.rewrite('plusminus') == (Jplus - Jminus) / (2 * I)
    assert represent(Jy, basis=Jz) == (represent(Jplus, basis=Jz) - represent(Jminus, basis=Jz)) / (2 * I)
    assert qapply(Jy * JxKet(1, 1)) == hbar * JxKet(1, 1)
    assert qapply(Jy * JyKet(1, 1)) == hbar * JyKet(1, 1)
    assert qapply(Jy * JzKet(1, 1)) == sqrt(2) * hbar * I * JzKet(1, 0) / 2
    assert qapply(Jy * JxKet(j, m)) == Sum(hbar * mi * WignerD(j, mi, m, pi * Rational(3, 2), 0, 0) * Sum(WignerD(j, mi1, mi, 0, 0, pi / 2) * JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jy * JyKet(j, m)) == hbar * m * JyKet(j, m)
    assert qapply(Jy * JzKet(j, m)) == -hbar * I * sqrt(j ** 2 + j - m ** 2 - m) * JzKet(j, m + 1) / 2 + hbar * I * sqrt(j ** 2 + j - m ** 2 + m) * JzKet(j, m - 1) / 2
    assert qapply(Jy * JxKetCoupled(1, 1, (1, 1))) == hbar * JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jy * JyKetCoupled(1, 1, (1, 1))) == hbar * JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jy * JzKetCoupled(1, 1, (1, 1))) == sqrt(2) * hbar * I * JzKetCoupled(1, 0, (1, 1)) / 2
    assert qapply(Jy * JxKetCoupled(j, m, (j1, j2))) == Sum(hbar * mi * WignerD(j, mi, m, pi * Rational(3, 2), 0, 0) * Sum(WignerD(j, mi1, mi, 0, 0, pi / 2) * JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jy * JyKetCoupled(j, m, (j1, j2))) == hbar * m * JyKetCoupled(j, m, (j1, j2))
    assert qapply(Jy * JzKetCoupled(j, m, (j1, j2))) == -hbar * I * sqrt(j ** 2 + j - m ** 2 - m) * JzKetCoupled(j, m + 1, (j1, j2)) / 2 + hbar * I * sqrt(j ** 2 + j - m ** 2 + m) * JzKetCoupled(j, m - 1, (j1, j2)) / 2
    assert qapply(Jy * TensorProduct(JxKet(1, 1), JxKet(1, 1))) == hbar * TensorProduct(JxKet(1, 1), JxKet(1, 1)) + hbar * TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert qapply(Jy * TensorProduct(JyKet(1, 1), JyKet(1, 1))) == 2 * hbar * TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert qapply(Jy * TensorProduct(JzKet(1, 1), JzKet(1, 1))) == sqrt(2) * hbar * I * TensorProduct(JzKet(1, 1), JzKet(1, 0)) / 2 + sqrt(2) * hbar * I * TensorProduct(JzKet(1, 0), JzKet(1, 1)) / 2
    assert qapply(Jy * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == 0
    assert qapply(Jy * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(JxKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, pi * Rational(3, 2), 0, 0) * Sum(WignerD(j2, mi1, mi, 0, 0, pi / 2) * JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2))) + TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, pi * Rational(3, 2), 0, 0) * Sum(WignerD(j1, mi1, mi, 0, 0, pi / 2) * JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(Jy * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == hbar * m1 * TensorProduct(JyKet(j1, m1), JyKet(j2, m2)) + hbar * m2 * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(Jy * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == -hbar * I * sqrt(j1 ** 2 + j1 - m1 ** 2 - m1) * TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2)) / 2 + hbar * I * sqrt(j1 ** 2 + j1 - m1 ** 2 + m1) * TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2)) / 2 + -hbar * I * sqrt(j2 ** 2 + j2 - m2 ** 2 - m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1)) / 2 + hbar * I * sqrt(j2 ** 2 + j2 - m2 ** 2 + m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1)) / 2
    assert qapply(TensorProduct(Jy, 1) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == hbar * TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jy) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == -hbar * TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(Jy, 1) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == hbar * TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jy) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == -hbar * TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(Jy, 1) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == hbar * sqrt(2) * I * TensorProduct(JzKet(1, 0), JzKet(1, -1)) / 2
    assert qapply(TensorProduct(1, Jy) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == -hbar * sqrt(2) * I * TensorProduct(JzKet(1, 1), JzKet(1, 0)) / 2
    assert qapply(TensorProduct(Jy, 1) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(Sum(hbar * mi * WignerD(j1, mi, m1, pi * Rational(3, 2), 0, 0) * Sum(WignerD(j1, mi1, mi, 0, 0, pi / 2) * JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jy) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(JxKet(j1, m1), Sum(hbar * mi * WignerD(j2, mi, m2, pi * Rational(3, 2), 0, 0) * Sum(WignerD(j2, mi1, mi, 0, 0, pi / 2) * JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jy, 1) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == hbar * m1 * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jy) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == hbar * m2 * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))
    assert qapply(TensorProduct(Jy, 1) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == -hbar * I * sqrt(j1 ** 2 + j1 - m1 ** 2 - m1) * TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2)) / 2 + hbar * I * sqrt(j1 ** 2 + j1 - m1 ** 2 + m1) * TensorProduct(JzKet(j1, m1 - 1), JzKet(j2, m2)) / 2
    assert qapply(TensorProduct(1, Jy) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == -hbar * I * sqrt(j2 ** 2 + j2 - m2 ** 2 - m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1)) / 2 + hbar * I * sqrt(j2 ** 2 + j2 - m2 ** 2 + m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 - 1)) / 2