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
def test_couple_symbolic():
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2))) == Sum(CG(j1, m1, j2, m2, j, m1 + m2) * JzKetCoupled(j, m1 + m2, (j1, j2)), (j, m1 + m2, j1 + j2))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3))) == Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j12, m1 + m2, j3, m3, j, m1 + m2 + m3) * JzKetCoupled(j, m1 + m2 + m3, (j1, j2, j3), ((1, 2, j12), (1, 3, j))), (j12, m1 + m2, j1 + j2), (j, m1 + m2 + m3, j12 + j3))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3)), ((1, 3), (1, 2))) == Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j2, m2, j, m1 + m2 + m3) * JzKetCoupled(j, m1 + m2 + m3, (j1, j2, j3), ((1, 3, j13), (1, 2, j))), (j13, m1 + m3, j1 + j3), (j, m1 + m2 + m3, j13 + j2))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4))) == Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j12, m1 + m2, j3, m3, j123, m1 + m2 + m3) * CG(j123, m1 + m2 + m3, j4, m4, j, m1 + m2 + m3 + m4) * JzKetCoupled(j, m1 + m2 + m3 + m4, (j1, j2, j3, j4), ((1, 2, j12), (1, 3, j123), (1, 4, j))), (j12, m1 + m2, j1 + j2), (j123, m1 + m2 + m3, j12 + j3), (j, m1 + m2 + m3 + m4, j123 + j4))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), ((1, 2), (3, 4), (1, 3))) == Sum(CG(j1, m1, j2, m2, j12, m1 + m2) * CG(j3, m3, j4, m4, j34, m3 + m4) * CG(j12, m1 + m2, j34, m3 + m4, j, m1 + m2 + m3 + m4) * JzKetCoupled(j, m1 + m2 + m3 + m4, (j1, j2, j3, j4), ((1, 2, j12), (3, 4, j34), (1, 3, j))), (j12, m1 + m2, j1 + j2), (j34, m3 + m4, j3 + j4), (j, m1 + m2 + m3 + m4, j12 + j34))
    assert couple(TensorProduct(JzKet(j1, m1), JzKet(j2, m2), JzKet(j3, m3), JzKet(j4, m4)), ((1, 3), (1, 4), (1, 2))) == Sum(CG(j1, m1, j3, m3, j13, m1 + m3) * CG(j13, m1 + m3, j4, m4, j134, m1 + m3 + m4) * CG(j134, m1 + m3 + m4, j2, m2, j, m1 + m2 + m3 + m4) * JzKetCoupled(j, m1 + m2 + m3 + m4, (j1, j2, j3, j4), ((1, 3, j13), (1, 4, j134), (1, 2, j))), (j13, m1 + m3, j1 + j3), (j134, m1 + m3 + m4, j13 + j4), (j, m1 + m2 + m3 + m4, j134 + j2))