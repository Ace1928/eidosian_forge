import itertools
import numpy as np
import pytest
import cirq
import cirq.testing
def test_sample_density_matrix():
    state = np.zeros(8, dtype=np.complex64)
    state[0] = 1 / np.sqrt(2)
    state[2] = 1 / np.sqrt(2)
    matrix = cirq.to_valid_density_matrix(state, num_qubits=3)
    for _ in range(10):
        sample = cirq.sample_density_matrix(matrix, [2, 1, 0])
        assert np.array_equal(sample, [[False, False, False]]) or np.array_equal(sample, [[False, True, False]])
    for _ in range(10):
        np.testing.assert_equal(cirq.sample_density_matrix(matrix, [2]), [[False]])
        np.testing.assert_equal(cirq.sample_density_matrix(matrix, [0]), [[False]])