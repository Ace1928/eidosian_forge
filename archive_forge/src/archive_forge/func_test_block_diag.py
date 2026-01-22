import pytest
import numpy as np
import cirq
def test_block_diag():
    assert np.allclose(cirq.block_diag(), np.zeros((0, 0)))
    assert np.allclose(cirq.block_diag(np.array([[1, 2], [3, 4]])), np.array([[1, 2], [3, 4]]))
    assert np.allclose(cirq.block_diag(np.array([[1, 2], [3, 4]]), np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]])), np.array([[1, 2, 0, 0, 0], [3, 4, 0, 0, 0], [0, 0, 4, 5, 6], [0, 0, 7, 8, 9], [0, 0, 10, 11, 12]]))