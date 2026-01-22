import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_no_repetitions():
    matrix = cirq.to_valid_density_matrix(0, 3)
    np.testing.assert_almost_equal(cirq.sample_density_matrix(matrix, [1], repetitions=0), np.zeros(shape=(0, 1)))
    np.testing.assert_almost_equal(cirq.sample_density_matrix(matrix, [0, 1], repetitions=0), np.zeros(shape=(0, 2)))