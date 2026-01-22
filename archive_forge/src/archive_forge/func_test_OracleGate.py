from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qubit import IntQubit
from sympy.physics.quantum.grover import (apply_grover, superposition_basis,
def test_OracleGate():
    v = OracleGate(1, lambda qubits: qubits == IntQubit(0))
    assert qapply(v * IntQubit(0)) == -IntQubit(0)
    assert qapply(v * IntQubit(1)) == IntQubit(1)
    nbits = 2
    v = OracleGate(2, return_one_on_two)
    assert qapply(v * IntQubit(0, nbits)) == IntQubit(0, nqubits=nbits)
    assert qapply(v * IntQubit(1, nbits)) == IntQubit(1, nqubits=nbits)
    assert qapply(v * IntQubit(2, nbits)) == -IntQubit(2, nbits)
    assert qapply(v * IntQubit(3, nbits)) == IntQubit(3, nbits)
    assert represent(OracleGate(1, lambda qubits: qubits == IntQubit(0)), nqubits=1) == Matrix([[-1, 0], [0, 1]])
    assert represent(v, nqubits=2) == Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])