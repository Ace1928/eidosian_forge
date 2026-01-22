import networkx as nx
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
@pytest.mark.parametrize('simplify', [False, True])
def test_circuit_to_tensors(simplify):
    rs = np.random.RandomState(52)
    qubits = cirq.LineQubit.range(2)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8)
    operator = cirq.PauliString({q: cirq.Z for q in rs.choice(qubits, size=2, replace=False)})
    circuit_sand = ccq.circuit_for_expectation_value(circuit, operator)
    if simplify:
        ccq.simplify_expectation_value_circuit(circuit_sand)
    qubits = sorted(circuit_sand.all_qubits())
    u_tn = ccq.tensor_unitary(circuit=circuit_sand, qubits=qubits)
    u_cirq = cirq.unitary(circuit_sand)
    np.testing.assert_allclose(u_tn, u_cirq, atol=1e-06)