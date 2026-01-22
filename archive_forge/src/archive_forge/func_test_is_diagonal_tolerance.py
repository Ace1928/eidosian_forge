import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_diagonal_tolerance():
    atol = 0.5
    assert cirq.is_diagonal(np.array([[1, 0], [-0.5, 1]]), atol=atol)
    assert not cirq.is_diagonal(np.array([[1, 0], [-0.6, 1]]), atol=atol)
    assert cirq.is_diagonal(np.array([[1, 0.5], [-0.5, 1]]), atol=atol)
    assert not cirq.is_diagonal(np.array([[1, 0.5], [-0.6, 1]]), atol=atol)