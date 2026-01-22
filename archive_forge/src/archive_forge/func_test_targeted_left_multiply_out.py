import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_targeted_left_multiply_out():
    left = np.array([[2, 3], [5, 7]])
    right = np.array([1, -1])
    out = np.zeros(2)
    result = cirq.targeted_left_multiply(left_matrix=left, right_target=right, target_axes=[0], out=out)
    assert result is out
    np.testing.assert_allclose(result, np.array([-1, -2]), atol=1e-08)