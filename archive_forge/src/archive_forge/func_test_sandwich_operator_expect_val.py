import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_sandwich_operator_expect_val():
    rs = np.random.RandomState(52)
    qubits = cirq.LineQubit.range(5)
    for _ in range(10):
        circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=10, op_density=0.8, random_state=rs)
        operator = _random_pauli_string(qubits, rs)
        tot_c = ccq.circuit_for_expectation_value(circuit, operator)
        eval_sandwich = cirq.unitary(tot_c)[0, 0]
        wfn = cirq.Simulator().simulate(circuit)
        eval_normal = operator.expectation_from_state_vector(wfn.final_state_vector, wfn.qubit_map)
        np.testing.assert_allclose(eval_sandwich, eval_normal, atol=1e-05)