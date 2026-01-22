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
def test_rewrite_Ket():
    assert JxKet(1, 1).rewrite('Jy') == I * JyKet(1, 1)
    assert JxKet(1, 0).rewrite('Jy') == JyKet(1, 0)
    assert JxKet(1, -1).rewrite('Jy') == -I * JyKet(1, -1)
    assert JxKet(1, 1).rewrite('Jz') == JzKet(1, 1) / 2 + JzKet(1, 0) / sqrt(2) + JzKet(1, -1) / 2
    assert JxKet(1, 0).rewrite('Jz') == -sqrt(2) * JzKet(1, 1) / 2 + sqrt(2) * JzKet(1, -1) / 2
    assert JxKet(1, -1).rewrite('Jz') == JzKet(1, 1) / 2 - JzKet(1, 0) / sqrt(2) + JzKet(1, -1) / 2
    assert JyKet(1, 1).rewrite('Jx') == -I * JxKet(1, 1)
    assert JyKet(1, 0).rewrite('Jx') == JxKet(1, 0)
    assert JyKet(1, -1).rewrite('Jx') == I * JxKet(1, -1)
    assert JyKet(1, 1).rewrite('Jz') == JzKet(1, 1) / 2 + sqrt(2) * I * JzKet(1, 0) / 2 - JzKet(1, -1) / 2
    assert JyKet(1, 0).rewrite('Jz') == sqrt(2) * I * JzKet(1, 1) / 2 + sqrt(2) * I * JzKet(1, -1) / 2
    assert JyKet(1, -1).rewrite('Jz') == -JzKet(1, 1) / 2 + sqrt(2) * I * JzKet(1, 0) / 2 + JzKet(1, -1) / 2
    assert JzKet(1, 1).rewrite('Jx') == JxKet(1, 1) / 2 - sqrt(2) * JxKet(1, 0) / 2 + JxKet(1, -1) / 2
    assert JzKet(1, 0).rewrite('Jx') == sqrt(2) * JxKet(1, 1) / 2 - sqrt(2) * JxKet(1, -1) / 2
    assert JzKet(1, -1).rewrite('Jx') == JxKet(1, 1) / 2 + sqrt(2) * JxKet(1, 0) / 2 + JxKet(1, -1) / 2
    assert JzKet(1, 1).rewrite('Jy') == JyKet(1, 1) / 2 - sqrt(2) * I * JyKet(1, 0) / 2 - JyKet(1, -1) / 2
    assert JzKet(1, 0).rewrite('Jy') == -sqrt(2) * I * JyKet(1, 1) / 2 - sqrt(2) * I * JyKet(1, -1) / 2
    assert JzKet(1, -1).rewrite('Jy') == -JyKet(1, 1) / 2 - sqrt(2) * I * JyKet(1, 0) / 2 + JyKet(1, -1) / 2
    assert JxKet(j, m).rewrite('Jy') == Sum(WignerD(j, mi, m, pi * Rational(3, 2), 0, 0) * JyKet(j, mi), (mi, -j, j))
    assert JxKet(j, m).rewrite('Jz') == Sum(WignerD(j, mi, m, 0, pi / 2, 0) * JzKet(j, mi), (mi, -j, j))
    assert JyKet(j, m).rewrite('Jx') == Sum(WignerD(j, mi, m, 0, 0, pi / 2) * JxKet(j, mi), (mi, -j, j))
    assert JyKet(j, m).rewrite('Jz') == Sum(WignerD(j, mi, m, pi * Rational(3, 2), -pi / 2, pi / 2) * JzKet(j, mi), (mi, -j, j))
    assert JzKet(j, m).rewrite('Jx') == Sum(WignerD(j, mi, m, 0, pi * Rational(3, 2), 0) * JxKet(j, mi), (mi, -j, j))
    assert JzKet(j, m).rewrite('Jy') == Sum(WignerD(j, mi, m, pi * Rational(3, 2), pi / 2, pi / 2) * JyKet(j, mi), (mi, -j, j))