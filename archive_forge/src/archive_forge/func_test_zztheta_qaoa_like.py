import pytest
import numpy as np
import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset
def test_zztheta_qaoa_like():
    qubits = cirq.LineQubit.range(4)
    for exponent in np.linspace(-1, 1, 10):
        circuit = cirq.Circuit([cirq.H.on_each(qubits), cirq.ZZPowGate(exponent=exponent)(qubits[0], qubits[1]), cirq.ZZPowGate(exponent=exponent)(qubits[2], qubits[3]), cirq.rx(0.123).on_each(qubits)])
        converted_circuit = cirq.optimize_for_target_gateset(circuit, gateset=cirq_google.SycamoreTargetGateset())
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(circuit, converted_circuit, atol=1e-08)