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
def test_rewrite_uncoupled_state():
    assert TensorProduct(JyKet(1, 1), JxKet(1, 1)).rewrite('Jx') == -I * TensorProduct(JxKet(1, 1), JxKet(1, 1))
    assert TensorProduct(JyKet(1, 0), JxKet(1, 1)).rewrite('Jx') == TensorProduct(JxKet(1, 0), JxKet(1, 1))
    assert TensorProduct(JyKet(1, -1), JxKet(1, 1)).rewrite('Jx') == I * TensorProduct(JxKet(1, -1), JxKet(1, 1))
    assert TensorProduct(JzKet(1, 1), JxKet(1, 1)).rewrite('Jx') == TensorProduct(JxKet(1, -1), JxKet(1, 1)) / 2 - sqrt(2) * TensorProduct(JxKet(1, 0), JxKet(1, 1)) / 2 + TensorProduct(JxKet(1, 1), JxKet(1, 1)) / 2
    assert TensorProduct(JzKet(1, 0), JxKet(1, 1)).rewrite('Jx') == -sqrt(2) * TensorProduct(JxKet(1, -1), JxKet(1, 1)) / 2 + sqrt(2) * TensorProduct(JxKet(1, 1), JxKet(1, 1)) / 2
    assert TensorProduct(JzKet(1, -1), JxKet(1, 1)).rewrite('Jx') == TensorProduct(JxKet(1, -1), JxKet(1, 1)) / 2 + sqrt(2) * TensorProduct(JxKet(1, 0), JxKet(1, 1)) / 2 + TensorProduct(JxKet(1, 1), JxKet(1, 1)) / 2
    assert TensorProduct(JxKet(1, 1), JyKet(1, 1)).rewrite('Jy') == I * TensorProduct(JyKet(1, 1), JyKet(1, 1))
    assert TensorProduct(JxKet(1, 0), JyKet(1, 1)).rewrite('Jy') == TensorProduct(JyKet(1, 0), JyKet(1, 1))
    assert TensorProduct(JxKet(1, -1), JyKet(1, 1)).rewrite('Jy') == -I * TensorProduct(JyKet(1, -1), JyKet(1, 1))
    assert TensorProduct(JzKet(1, 1), JyKet(1, 1)).rewrite('Jy') == -TensorProduct(JyKet(1, -1), JyKet(1, 1)) / 2 - sqrt(2) * I * TensorProduct(JyKet(1, 0), JyKet(1, 1)) / 2 + TensorProduct(JyKet(1, 1), JyKet(1, 1)) / 2
    assert TensorProduct(JzKet(1, 0), JyKet(1, 1)).rewrite('Jy') == -sqrt(2) * I * TensorProduct(JyKet(1, -1), JyKet(1, 1)) / 2 - sqrt(2) * I * TensorProduct(JyKet(1, 1), JyKet(1, 1)) / 2
    assert TensorProduct(JzKet(1, -1), JyKet(1, 1)).rewrite('Jy') == TensorProduct(JyKet(1, -1), JyKet(1, 1)) / 2 - sqrt(2) * I * TensorProduct(JyKet(1, 0), JyKet(1, 1)) / 2 - TensorProduct(JyKet(1, 1), JyKet(1, 1)) / 2
    assert TensorProduct(JxKet(1, 1), JzKet(1, 1)).rewrite('Jz') == TensorProduct(JzKet(1, -1), JzKet(1, 1)) / 2 + sqrt(2) * TensorProduct(JzKet(1, 0), JzKet(1, 1)) / 2 + TensorProduct(JzKet(1, 1), JzKet(1, 1)) / 2
    assert TensorProduct(JxKet(1, 0), JzKet(1, 1)).rewrite('Jz') == sqrt(2) * TensorProduct(JzKet(1, -1), JzKet(1, 1)) / 2 - sqrt(2) * TensorProduct(JzKet(1, 1), JzKet(1, 1)) / 2
    assert TensorProduct(JxKet(1, -1), JzKet(1, 1)).rewrite('Jz') == TensorProduct(JzKet(1, -1), JzKet(1, 1)) / 2 - sqrt(2) * TensorProduct(JzKet(1, 0), JzKet(1, 1)) / 2 + TensorProduct(JzKet(1, 1), JzKet(1, 1)) / 2
    assert TensorProduct(JyKet(1, 1), JzKet(1, 1)).rewrite('Jz') == -TensorProduct(JzKet(1, -1), JzKet(1, 1)) / 2 + sqrt(2) * I * TensorProduct(JzKet(1, 0), JzKet(1, 1)) / 2 + TensorProduct(JzKet(1, 1), JzKet(1, 1)) / 2
    assert TensorProduct(JyKet(1, 0), JzKet(1, 1)).rewrite('Jz') == sqrt(2) * I * TensorProduct(JzKet(1, -1), JzKet(1, 1)) / 2 + sqrt(2) * I * TensorProduct(JzKet(1, 1), JzKet(1, 1)) / 2
    assert TensorProduct(JyKet(1, -1), JzKet(1, 1)).rewrite('Jz') == TensorProduct(JzKet(1, -1), JzKet(1, 1)) / 2 + sqrt(2) * I * TensorProduct(JzKet(1, 0), JzKet(1, 1)) / 2 - TensorProduct(JzKet(1, 1), JzKet(1, 1)) / 2
    assert TensorProduct(JyKet(j1, m1), JxKet(j2, m2)).rewrite('Jy') == TensorProduct(JyKet(j1, m1), Sum(WignerD(j2, mi, m2, pi * Rational(3, 2), 0, 0) * JyKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JzKet(j1, m1), JxKet(j2, m2)).rewrite('Jz') == TensorProduct(JzKet(j1, m1), Sum(WignerD(j2, mi, m2, 0, pi / 2, 0) * JzKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JxKet(j1, m1), JyKet(j2, m2)).rewrite('Jx') == TensorProduct(JxKet(j1, m1), Sum(WignerD(j2, mi, m2, 0, 0, pi / 2) * JxKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JzKet(j1, m1), JyKet(j2, m2)).rewrite('Jz') == TensorProduct(JzKet(j1, m1), Sum(WignerD(j2, mi, m2, pi * Rational(3, 2), -pi / 2, pi / 2) * JzKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JxKet(j1, m1), JzKet(j2, m2)).rewrite('Jx') == TensorProduct(JxKet(j1, m1), Sum(WignerD(j2, mi, m2, 0, pi * Rational(3, 2), 0) * JxKet(j2, mi), (mi, -j2, j2)))
    assert TensorProduct(JyKet(j1, m1), JzKet(j2, m2)).rewrite('Jy') == TensorProduct(JyKet(j1, m1), Sum(WignerD(j2, mi, m2, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j2, mi), (mi, -j2, j2)))