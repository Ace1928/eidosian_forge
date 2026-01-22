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
def test_jplus():
    assert Commutator(Jplus, Jminus).doit() == 2 * hbar * Jz
    assert Jplus.matrix_element(1, 1, 1, 1) == 0
    assert Jplus.rewrite('xyz') == Jx + I * Jy
    assert qapply(Jplus * JxKet(1, 1)) == -hbar * sqrt(2) * JxKet(1, 0) / 2 + hbar * JxKet(1, 1)
    assert qapply(Jplus * JyKet(1, 1)) == hbar * sqrt(2) * JyKet(1, 0) / 2 + I * hbar * JyKet(1, 1)
    assert qapply(Jplus * JzKet(1, 1)) == 0
    assert qapply(Jplus * JxKet(j, m)) == Sum(hbar * sqrt(-mi ** 2 - mi + j ** 2 + j) * WignerD(j, mi, m, 0, pi / 2, 0) * Sum(WignerD(j, mi1, mi + 1, 0, pi * Rational(3, 2), 0) * JxKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus * JyKet(j, m)) == Sum(hbar * sqrt(j ** 2 + j - mi ** 2 - mi) * WignerD(j, mi, m, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j, mi1, mi + 1, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j, mi1), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus * JzKet(j, m)) == hbar * sqrt(j ** 2 + j - m ** 2 - m) * JzKet(j, m + 1)
    assert qapply(Jplus * JxKetCoupled(1, 1, (1, 1))) == -hbar * sqrt(2) * JxKetCoupled(1, 0, (1, 1)) / 2 + hbar * JxKetCoupled(1, 1, (1, 1))
    assert qapply(Jplus * JyKetCoupled(1, 1, (1, 1))) == hbar * sqrt(2) * JyKetCoupled(1, 0, (1, 1)) / 2 + I * hbar * JyKetCoupled(1, 1, (1, 1))
    assert qapply(Jplus * JzKet(1, 1)) == 0
    assert qapply(Jplus * JxKetCoupled(j, m, (j1, j2))) == Sum(hbar * sqrt(-mi ** 2 - mi + j ** 2 + j) * WignerD(j, mi, m, 0, pi / 2, 0) * Sum(WignerD(j, mi1, mi + 1, 0, pi * Rational(3, 2), 0) * JxKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus * JyKetCoupled(j, m, (j1, j2))) == Sum(hbar * sqrt(j ** 2 + j - mi ** 2 - mi) * WignerD(j, mi, m, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j, mi1, mi + 1, pi * Rational(3, 2), pi / 2, pi / 2) * JyKetCoupled(j, mi1, (j1, j2)), (mi1, -j, j)), (mi, -j, j))
    assert qapply(Jplus * JzKetCoupled(j, m, (j1, j2))) == hbar * sqrt(j ** 2 + j - m ** 2 - m) * JzKetCoupled(j, m + 1, (j1, j2))
    assert qapply(TensorProduct(Jplus, 1) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == -hbar * sqrt(2) * TensorProduct(JxKet(1, 0), JxKet(1, -1)) / 2 + hbar * TensorProduct(JxKet(1, 1), JxKet(1, -1))
    assert qapply(TensorProduct(1, Jplus) * TensorProduct(JxKet(1, 1), JxKet(1, -1))) == -hbar * TensorProduct(JxKet(1, 1), JxKet(1, -1)) + hbar * sqrt(2) * TensorProduct(JxKet(1, 1), JxKet(1, 0)) / 2
    assert qapply(TensorProduct(Jplus, 1) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == hbar * sqrt(2) * TensorProduct(JyKet(1, 0), JyKet(1, -1)) / 2 + hbar * I * TensorProduct(JyKet(1, 1), JyKet(1, -1))
    assert qapply(TensorProduct(1, Jplus) * TensorProduct(JyKet(1, 1), JyKet(1, -1))) == -hbar * I * TensorProduct(JyKet(1, 1), JyKet(1, -1)) + hbar * sqrt(2) * TensorProduct(JyKet(1, 1), JyKet(1, 0)) / 2
    assert qapply(TensorProduct(Jplus, 1) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == 0
    assert qapply(TensorProduct(1, Jplus) * TensorProduct(JzKet(1, 1), JzKet(1, -1))) == hbar * sqrt(2) * TensorProduct(JzKet(1, 1), JzKet(1, 0))
    assert qapply(TensorProduct(Jplus, 1) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(Sum(hbar * sqrt(-mi ** 2 - mi + j1 ** 2 + j1) * WignerD(j1, mi, m1, 0, pi / 2, 0) * Sum(WignerD(j1, mi1, mi + 1, 0, pi * Rational(3, 2), 0) * JxKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JxKet(j2, m2))
    assert qapply(TensorProduct(1, Jplus) * TensorProduct(JxKet(j1, m1), JxKet(j2, m2))) == TensorProduct(JxKet(j1, m1), Sum(hbar * sqrt(-mi ** 2 - mi + j2 ** 2 + j2) * WignerD(j2, mi, m2, 0, pi / 2, 0) * Sum(WignerD(j2, mi1, mi + 1, 0, pi * Rational(3, 2), 0) * JxKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jplus, 1) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(Sum(hbar * sqrt(j1 ** 2 + j1 - mi ** 2 - mi) * WignerD(j1, mi, m1, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j1, mi1, mi + 1, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j1, mi1), (mi1, -j1, j1)), (mi, -j1, j1)), JyKet(j2, m2))
    assert qapply(TensorProduct(1, Jplus) * TensorProduct(JyKet(j1, m1), JyKet(j2, m2))) == TensorProduct(JyKet(j1, m1), Sum(hbar * sqrt(j2 ** 2 + j2 - mi ** 2 - mi) * WignerD(j2, mi, m2, pi * Rational(3, 2), -pi / 2, pi / 2) * Sum(WignerD(j2, mi1, mi + 1, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j2, mi1), (mi1, -j2, j2)), (mi, -j2, j2)))
    assert qapply(TensorProduct(Jplus, 1) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * sqrt(j1 ** 2 + j1 - m1 ** 2 - m1) * TensorProduct(JzKet(j1, m1 + 1), JzKet(j2, m2))
    assert qapply(TensorProduct(1, Jplus) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == hbar * sqrt(j2 ** 2 + j2 - m2 ** 2 - m2) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2 + 1))