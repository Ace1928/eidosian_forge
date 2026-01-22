import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('num_controls', [*range(7, 17)])
@pytest.mark.parametrize('pauli', [cirq.X, cirq.Y, cirq.Z])
@pytest.mark.parametrize('cv', [0, 1])
@allow_deprecated_cirq_ft_use_in_tests
def test_t_complexity_mcp(num_controls: int, pauli: cirq.Pauli, cv: int):
    gate = cirq_ft.MultiControlPauli([cv] * num_controls, target_gate=pauli)
    cirq_ft.testing.assert_decompose_is_consistent_with_t_complexity(gate)