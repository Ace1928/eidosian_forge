import random
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_partial,
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_measure_normalize():
    a, b = symbols('a b')
    state = a * Qubit('110') + b * Qubit('111')
    assert measure_partial(state, (0,), normalize=False) == [(a * Qubit('110'), a * a.conjugate()), (b * Qubit('111'), b * b.conjugate())]
    assert measure_all(state, normalize=False) == [(Qubit('110'), a * a.conjugate()), (Qubit('111'), b * b.conjugate())]