from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer, Rational, pi)
from sympy.core.symbol import (Wild, symbols)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices import Matrix, ImmutableMatrix
from sympy.physics.quantum.gate import (XGate, YGate, ZGate, random_circuit,
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import Qubit, IntQubit, qubit_to_matrix, \
from sympy.physics.quantum.matrixutils import matrix_to_zero
from sympy.physics.quantum.matrixcache import sqrt2_inv
from sympy.physics.quantum import Dagger
def test_UGate():
    a, b, c, d = symbols('a,b,c,d')
    uMat = Matrix([[a, b], [c, d]])
    u1 = UGate((0,), uMat)
    assert represent(u1, nqubits=1) == uMat
    assert qapply(u1 * Qubit('0')) == a * Qubit('0') + c * Qubit('1')
    assert qapply(u1 * Qubit('1')) == b * Qubit('0') + d * Qubit('1')
    u2 = UGate((1,), uMat)
    u2Rep = represent(u2, nqubits=2)
    for i in range(4):
        assert u2Rep * qubit_to_matrix(IntQubit(i, 2)) == qubit_to_matrix(qapply(u2 * IntQubit(i, 2)))