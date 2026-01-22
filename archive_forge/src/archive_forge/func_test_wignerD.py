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
def test_wignerD():
    i, j = symbols('i j')
    assert Rotation.D(1, 1, 1, 0, 0, 0) == WignerD(1, 1, 1, 0, 0, 0)
    assert Rotation.D(1, 1, 2, 0, 0, 0) == WignerD(1, 1, 2, 0, 0, 0)
    assert Rotation.D(1, i ** 2 - j ** 2, i ** 2 - j ** 2, 0, 0, 0) == WignerD(1, i ** 2 - j ** 2, i ** 2 - j ** 2, 0, 0, 0)
    assert Rotation.D(1, i, i, 0, 0, 0) == WignerD(1, i, i, 0, 0, 0)
    assert Rotation.D(1, i, i + 1, 0, 0, 0) == WignerD(1, i, i + 1, 0, 0, 0)
    assert Rotation.D(1, 0, 0, 0, 0, 0) == WignerD(1, 0, 0, 0, 0, 0)