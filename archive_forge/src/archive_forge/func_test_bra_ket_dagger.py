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
def test_bra_ket_dagger():
    x = symbols('x', complex=True)
    k = Ket('k')
    b = Bra('b')
    assert Dagger(k) == Bra('k')
    assert Dagger(b) == Ket('b')
    assert Dagger(k).is_commutative is False
    k2 = Ket('k2')
    e = 2 * I * k + x * k2
    assert Dagger(e) == conjugate(x) * Dagger(k2) - 2 * I * Dagger(k)