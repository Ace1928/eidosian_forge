from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
@pytest.mark.parametrize('theta', np.linspace(0, 2 * np.pi, 10))
@pytest.mark.parametrize('phase_state', [0, 1])
@pytest.mark.parametrize('target_bitsize', [1, 2, 3])
@pytest.mark.parametrize('ancilla_bitsize', [1, 4])
def test_decompose_gate_that_allocates_clean_qubits(theta: float, phase_state: int, target_bitsize: int, ancilla_bitsize: int):
    gate = testing.PhaseUsingCleanAncilla(theta, phase_state, target_bitsize, ancilla_bitsize)
    _test_gate_that_allocates_qubits(gate)