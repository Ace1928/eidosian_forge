import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('matrix', [np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 5]]), np.array([[1, 1], [1, 1]]), np.array([[0, 1], [1, 0]]), np.array([[0, 2], [2, 0]]), np.array([[-1, 500], [500, 4]]), np.array([[-1, 500], [500, -4]]), np.array([[1, 3], [3, 7]])] + [random_symmetric_matrix(2) for _ in range(10)] + [random_symmetric_matrix(4) for _ in range(10)] + [random_symmetric_matrix(k) for k in range(1, 10)])
def test_diagonalize_real_symmetric_matrix(matrix):
    p = cirq.diagonalize_real_symmetric_matrix(matrix)
    assert_diagonalized_by(matrix, p)