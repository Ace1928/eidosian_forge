import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_convert_to_sycamore_tabulation():
    sycamore_tabulation = cirq.two_qubit_gate_product_tabulation(cirq.unitary(cirq_google.SYC), 0.1, random_state=cirq.value.parse_random_state(11))
    circuit = cirq.Circuit(cirq.MatrixGate(cirq.unitary(cirq.CX)).on(*cirq.LineQubit.range(2)))
    converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset(tabulation=sycamore_tabulation))
    u1 = cirq.unitary(circuit)
    u2 = cirq.unitary(converted_circuit)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0, 0.1)