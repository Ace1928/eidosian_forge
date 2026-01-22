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
def test_ket():
    k = Ket('0')
    assert isinstance(k, Ket)
    assert isinstance(k, KetBase)
    assert isinstance(k, StateBase)
    assert isinstance(k, QExpr)
    assert k.label == (Symbol('0'),)
    assert k.hilbert_space == HilbertSpace()
    assert k.is_commutative is False
    k = Ket('pi')
    assert k.label == (Symbol('pi'),)
    k = Ket(x, y)
    assert k.label == (x, y)
    assert k.hilbert_space == HilbertSpace()
    assert k.is_commutative is False
    assert k.dual_class() == Bra
    assert k.dual == Bra(x, y)
    assert k.subs(x, y) == Ket(y, y)
    k = CustomKet()
    assert k == CustomKet('test')
    k = CustomKetMultipleLabels()
    assert k == CustomKetMultipleLabels('r', 'theta', 'phi')
    assert Ket() == Ket('psi')