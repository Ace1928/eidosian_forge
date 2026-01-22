import cmath
import numpy as np
import pytest
import cirq
from cirq.linalg import matrix_commutes
def test_is_normal_tolerance():
    atol = 0.25
    assert cirq.is_normal(np.array([[0, 0.5], [0, 0]]), atol=atol)
    assert not cirq.is_normal(np.array([[0, 0.6], [0, 0]]), atol=atol)
    assert cirq.is_normal(np.array([[0, 0.5, 0], [0, 0, 0.5], [0, 0, 0]]), atol=atol)
    assert not cirq.is_normal(np.array([[0, 0.5, 0], [0, 0, 0.6], [0, 0, 0]]), atol=atol)