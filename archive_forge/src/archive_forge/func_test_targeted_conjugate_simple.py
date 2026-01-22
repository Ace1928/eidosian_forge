import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
def test_targeted_conjugate_simple():
    a = np.array([[0, 1j], [0, 0]])
    b = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]), (2,) * 4)
    expected = np.reshape(np.array([11, 12, 0, 0, 15, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), (2,) * 4)
    result = cirq.targeted_conjugate_about(a, b, [0])
    np.testing.assert_almost_equal(result, expected)