import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_state_empty_density_matrix():
    matrix = np.zeros(shape=())
    bits, out_matrix = cirq.measure_density_matrix(matrix, [])
    assert [] == bits
    np.testing.assert_almost_equal(matrix, out_matrix)