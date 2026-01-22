import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix_big_endian():
    results = []
    for x in range(8):
        matrix = cirq.to_valid_density_matrix(x, 3)
        sample = cirq.sample_density_matrix(matrix, [2, 1, 0])
        results.append(sample)
    expecteds = [[list(reversed(x))] for x in list(itertools.product([False, True], repeat=3))]
    for result, expected in zip(results, expecteds):
        np.testing.assert_equal(result, expected)