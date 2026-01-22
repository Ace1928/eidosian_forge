import random
import numpy as np
import pytest
import cirq
from cirq import value
from cirq import unitary_eig
@pytest.mark.parametrize('matrix,exponent,desired', [[X, 2, np.eye(2)], [X, 3, X], [Z, 2, np.eye(2)], [H, 2, np.eye(2)], [Z, 0.5, np.diag([1, 1j])], [X, 0.5, np.array([[1j, 1], [1, 1j]]) * (1 - 1j) / 2]])
def test_map_eigenvalues_raise(matrix, exponent, desired):
    exp_mapped = cirq.map_eigenvalues(matrix, lambda e: complex(e) ** exponent)
    assert np.allclose(desired, exp_mapped)