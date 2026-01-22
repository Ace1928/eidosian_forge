import pytest
import numpy as np
import cirq
from cirq.testing import assert_allclose_up_to_global_phase
def test_clifford_decompose_two_qubits():
    """Two random instance for two qubits decomposition."""
    qubits = cirq.LineQubit.range(2)
    args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=2), qubits=qubits, prng=np.random.RandomState())
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CNOT, args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    expect_circ = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]))
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-07)
    qubits = cirq.LineQubit.range(2)
    args = cirq.CliffordTableauSimulationState(tableau=cirq.CliffordTableau(num_qubits=2), qubits=qubits, prng=np.random.RandomState())
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CNOT, args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    cirq.act_on(cirq.H, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.S, args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.X, args, qubits=[qubits[1]], allow_decompose=False)
    expect_circ = cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.H(qubits[0]), cirq.S(qubits[0]), cirq.X(qubits[1]))
    ops = cirq.decompose_clifford_tableau_to_operations(qubits, args.tableau)
    circ = cirq.Circuit(ops)
    assert_allclose_up_to_global_phase(cirq.unitary(expect_circ), cirq.unitary(circ), atol=1e-07)