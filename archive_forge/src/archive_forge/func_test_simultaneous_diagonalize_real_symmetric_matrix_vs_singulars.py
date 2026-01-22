import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('s,m', [([1, 1], np.eye(2)), ([-1, -1], np.eye(2)), ([2, 1], np.eye(2)), ([2, 0], np.eye(2)), ([0, 0], np.eye(2)), ([1, 1], [[0, 1], [1, 0]]), ([2, 2], [[0, 1], [1, 0]]), ([1, 1], [[1, 3], [3, 6]]), ([2, 2, 1], [[1, 3, 0], [3, 6, 0], [0, 0, 1]]), ([2, 1, 1], [[-5, 0, 0], [0, 1, 3], [0, 3, 6]])] + [([6, 6, 5, 5, 5], random_block_diagonal_symmetric_matrix(2, 3)) for _ in range(10)])
def test_simultaneous_diagonalize_real_symmetric_matrix_vs_singulars(s, m):
    m = np.array(m)
    s = np.diag(s)
    p = cirq.diagonalize_real_symmetric_and_sorted_diagonal_matrices(m, s)
    assert_diagonalized_by(s, p)
    assert_diagonalized_by(m, p)
    assert np.allclose(s, p.T.dot(s).dot(p))