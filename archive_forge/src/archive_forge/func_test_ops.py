from sympy.core.add import Add
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, oo, pi)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import sympify
from sympy.functions.elementary.complexes import conjugate
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin
from sympy.testing.pytest import raises
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.state import (
from sympy.physics.quantum.hilbert import HilbertSpace
def test_ops():
    k0 = Ket(0)
    k1 = Ket(1)
    k = 2 * I * k0 - x / sqrt(2) * k1
    assert k == Add(Mul(2, I, k0), Mul(Rational(-1, 2), x, Pow(2, S.Half), k1))