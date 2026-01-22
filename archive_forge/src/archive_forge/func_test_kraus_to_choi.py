from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('kraus_operators, expected_choi', (([np.eye(2)], np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]])), (cirq.kraus(cirq.depolarize(0.75)), np.eye(4) / 2), ([np.array([[1, 0, 0], [0, 0, 1]]) / np.sqrt(2), np.array([[1, 0, 0], [0, 0, -1]]) / np.sqrt(2)], np.diag([1, 0, 0, 0, 0, 1]))))
def test_kraus_to_choi(kraus_operators, expected_choi):
    """Verifies that cirq.kraus_to_choi computes the correct Choi matrix."""
    assert np.allclose(cirq.kraus_to_choi(kraus_operators), expected_choi)