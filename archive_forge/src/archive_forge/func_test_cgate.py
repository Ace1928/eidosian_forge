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
def test_cgate():
    """Test the general CGate."""
    CNOTMatrix = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert represent(CGate(1, XGate(0)), nqubits=2) == CNOTMatrix
    ToffoliGate = CGate((1, 2), XGate(0))
    assert represent(ToffoliGate, nqubits=3) == Matrix([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 1, 0]])
    ToffoliGate = CGate((3, 0), XGate(1))
    assert qapply(ToffoliGate * Qubit('1001')) == matrix_to_qubit(represent(ToffoliGate * Qubit('1001'), nqubits=4))
    assert qapply(ToffoliGate * Qubit('0000')) == matrix_to_qubit(represent(ToffoliGate * Qubit('0000'), nqubits=4))
    CYGate = CGate(1, YGate(0))
    CYGate_matrix = Matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, -I), (0, 0, I, 0)))
    assert represent(CYGate.decompose(), nqubits=2) == CYGate_matrix
    CZGate = CGate(0, ZGate(1))
    CZGate_matrix = Matrix(((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, -1)))
    assert qapply(CZGate * Qubit('11')) == -Qubit('11')
    assert matrix_to_qubit(represent(CZGate * Qubit('11'), nqubits=2)) == -Qubit('11')
    assert represent(CZGate.decompose(), nqubits=2) == CZGate_matrix
    CPhaseGate = CGate(0, PhaseGate(1))
    assert qapply(CPhaseGate * Qubit('11')) == I * Qubit('11')
    assert matrix_to_qubit(represent(CPhaseGate * Qubit('11'), nqubits=2)) == I * Qubit('11')
    assert Dagger(CZGate) == CZGate
    assert pow(CZGate, 1) == Dagger(CZGate)
    assert Dagger(CZGate) == CZGate.inverse()
    assert Dagger(CPhaseGate) != CPhaseGate
    assert Dagger(CPhaseGate) == CPhaseGate.inverse()
    assert Dagger(CPhaseGate) == pow(CPhaseGate, -1)
    assert pow(CPhaseGate, -1) == CPhaseGate.inverse()