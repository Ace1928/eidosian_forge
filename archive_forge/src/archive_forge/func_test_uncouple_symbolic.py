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
def test_uncouple_symbolic():
    assert uncouple(JzKetCoupled(j, m, (j1, j2))) == Sum(CG(j1, m1, j2, m2, j, m) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2)), (m1, -j1, j1), (m2, -j2, j2))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3))) == Sum(CG(j1, m1, j2, m2, j1 + j2, m1 + m2) * CG(j1 + j2, m1 + m2, j3, m3, j, m) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)), (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3), ((1, 3, j13), (1, 2, j)))) == Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j2, m2, j, m) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)), (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3, j4))) == Sum(CG(j1, m1, j2, m2, j1 + j2, m1 + m2) * CG(j1 + j2, m1 + m2, j3, m3, j1 + j2 + j3, m1 + m2 + m3) * CG(j1 + j2 + j3, m1 + m2 + m3, j4, m4, j, m) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3), (m4, -j4, j4))
    assert uncouple(JzKetCoupled(j, m, (j1, j2, j3, j4), ((1, 3, j13), (2, 4, j24), (1, 2, j)))) == Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j2, m2, j4, m4, j24, m2 + m4) * CG(j13, m1 + m3, j24, m2 + m4, j, m) * TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), (m1, -j1, j1), (m2, -j2, j2), (m3, -j3, j3), (m4, -j4, j4))