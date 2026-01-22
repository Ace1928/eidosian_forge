import cirq
import cirq_ionq as ionq
import pytest
import sympy
def test_decompose_single_qubit_matrix_gate():
    q = cirq.LineQubit(0)
    for _ in range(10):
        gate = cirq.MatrixGate(cirq.testing.random_unitary(2))
        circuit = cirq.Circuit(gate(q))
        decomposed_circuit = cirq.optimize_for_target_gateset(circuit, gateset=ionq_target_gateset, ignore_failures=False)
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, decomposed_circuit, atol=1e-08)
        assert VALID_DECOMPOSED_GATES.validate(decomposed_circuit)