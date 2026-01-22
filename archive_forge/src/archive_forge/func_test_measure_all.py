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
def test_measure_all():
    assert measure_all(Qubit('11')) == [(Qubit('11'), 1)]
    state = Qubit('11') + Qubit('10')
    assert measure_all(state) == [(Qubit('10'), S.Half), (Qubit('11'), S.Half)]
    state2 = Qubit('11') / sqrt(5) + 2 * Qubit('00') / sqrt(5)
    assert measure_all(state2) == [(Qubit('00'), Rational(4, 5)), (Qubit('11'), Rational(1, 5))]
    assert measure_all(qapply(Qubit('0'))) == [(Qubit('0'), 1)]