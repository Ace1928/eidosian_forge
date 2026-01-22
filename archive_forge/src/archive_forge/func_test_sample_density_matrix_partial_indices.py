import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_partial_indices():
    for index in range(3):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            np.testing.assert_equal(cirq.sample_density_matrix(matrix, [index]), [[bool(1 & x >> 2 - index)]])