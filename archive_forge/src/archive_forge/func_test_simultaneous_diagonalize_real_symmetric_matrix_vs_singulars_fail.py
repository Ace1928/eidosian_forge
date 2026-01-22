import random
from typing import Tuple, Optional
import numpy as np
import pytest
import cirq
@pytest.mark.parametrize('s,m,match', [([1, 2], np.eye(2), 'must be real diagonal descending'), ([2, 1], [[0, 1], [1, 0]], 'must commute'), ([1, 0], [[1, 3], [3, 6]], 'must commute'), ([2, 1, 1], [[1, 3, 0], [3, 6, 0], [0, 0, 1]], 'must commute'), ([2, 2, 1], [[-5, 0, 0], [0, 1, 3], [0, 3, 6]], 'must commute'), ([3, 2, 1], QFT, 'must be real symmetric')])
def test_simultaneous_diagonalize_real_symmetric_matrix_vs_singulars_fail(s, m, match: str):
    m = np.array(m)
    s = np.diag(s)
    with pytest.raises(ValueError, match=match):
        cirq.diagonalize_real_symmetric_and_sorted_diagonal_matrices(m, s)