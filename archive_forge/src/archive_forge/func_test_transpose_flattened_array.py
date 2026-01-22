import numpy as np
import pytest
import cirq
import cirq.testing
from cirq import linalg
@pytest.mark.parametrize('num_dimensions', [*range(1, 7)])
def test_transpose_flattened_array(num_dimensions):
    np.random.seed(0)
    for _ in range(10):
        shape = np.random.randint(1, 5, (num_dimensions,)).tolist()
        axes = np.random.permutation(num_dimensions).tolist()
        volume = np.prod(shape)
        A = np.random.permutation(volume)
        want = np.transpose(A.reshape(shape), axes)
        got = linalg.transpose_flattened_array(A, shape, axes).reshape(want.shape)
        assert np.array_equal(want, got)
        got = linalg.transpose_flattened_array(A.reshape(shape), shape, axes).reshape(want.shape)
        assert np.array_equal(want, got)