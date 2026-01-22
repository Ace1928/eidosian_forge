import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
@pytest.mark.parametrize('pauli_error,num_qubits,expected_output', [(0.01, 1, 1 - 0.01 / (3 / 4)), (0.05, 2, 1 - 0.05 / (15 / 16))])
def test_pauli_error_to_decay_constant(pauli_error, num_qubits, expected_output):
    val = pauli_error_to_decay_constant(pauli_error, num_qubits)
    assert val == expected_output