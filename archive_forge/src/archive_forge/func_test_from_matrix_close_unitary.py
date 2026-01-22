import random
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('unitary', [cirq.testing.random_unitary(2), cirq.testing.random_unitary(2), cirq.testing.random_unitary(2), np.array([[0, 1], [1j, 0]])])
def test_from_matrix_close_unitary(unitary: np.ndarray):
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(cirq.PhasedXZGate.from_matrix(unitary)), unitary, atol=1e-08)