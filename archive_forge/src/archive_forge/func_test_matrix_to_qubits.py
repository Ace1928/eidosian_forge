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
def test_matrix_to_qubits():
    qb = Qubit(0, 0, 0, 0)
    mat = Matrix([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert matrix_to_qubit(mat) == qb
    assert qubit_to_matrix(qb) == mat
    state = 2 * sqrt(2) * (Qubit(0, 0, 0) + Qubit(0, 0, 1) + Qubit(0, 1, 0) + Qubit(0, 1, 1) + Qubit(1, 0, 0) + Qubit(1, 0, 1) + Qubit(1, 1, 0) + Qubit(1, 1, 1))
    ones = sqrt(2) * 2 * Matrix([1, 1, 1, 1, 1, 1, 1, 1])
    assert matrix_to_qubit(ones) == state.expand()
    assert qubit_to_matrix(state) == ones