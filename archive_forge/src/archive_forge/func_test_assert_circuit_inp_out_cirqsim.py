import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
def test_assert_circuit_inp_out_cirqsim():
    qubits = cirq.LineQubit.range(4)
    initial_state = [0, 1, 0, 0]
    circuit = cirq.Circuit(cirq.X(qubits[3]))
    final_state = [0, 1, 0, 1]
    cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)
    final_state = [0, 0, 0, 1]
    with pytest.raises(AssertionError):
        cirq_ft.testing.assert_circuit_inp_out_cirqsim(circuit, qubits, initial_state, final_state)