import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_state_out_is_not_matrix():
    matrix = matrix_000_plus_010()
    out = np.zeros_like(matrix)
    _, out_matrix = cirq.measure_density_matrix(matrix, [2, 1, 0], out=out)
    assert out is not matrix
    assert out is out_matrix