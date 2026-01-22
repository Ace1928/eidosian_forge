from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('superoperator', (np.eye(4), np.diag([1, 0, 0, 1]), np.diag([1, -1j, 1j, 1]), np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]), np.array([[1, 0, 0, 0.8], [0, 0.36, 0, 0], [0, 0, 0.36, 0], [0, 0, 0, 0.64]])))
def test_superoperator_to_kraus_inverse_of_kraus_to_superoperator(superoperator):
    """Verifies that cirq.kraus_to_superoperator(cirq.superoperator_to_kraus(.)) is identity."""
    kraus = cirq.superoperator_to_kraus(superoperator)
    recovered_superoperator = cirq.kraus_to_superoperator(kraus)
    assert np.allclose(recovered_superoperator, superoperator)