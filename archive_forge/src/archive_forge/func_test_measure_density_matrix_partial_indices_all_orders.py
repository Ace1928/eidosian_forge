import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_measure_density_matrix_partial_indices_all_orders():
    for perm in itertools.permutations([0, 1, 2]):
        for x in range(8):
            matrix = cirq.to_valid_density_matrix(x, 3)
            bits, out_matrix = cirq.measure_density_matrix(matrix, perm)
            np.testing.assert_almost_equal(matrix, out_matrix)
            assert bits == [bool(1 & x >> 2 - p) for p in perm]