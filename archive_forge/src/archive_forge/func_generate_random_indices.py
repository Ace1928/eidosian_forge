import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def generate_random_indices(self):
    N = min(self.shape)
    slice_choices = [slice(None, None, None), slice(1, N - 1, None), slice(0, None, 2), slice(N - 1, None, -2), slice(-N + 1, -1, None), slice(-1, -N, -2), slice(0, N - 1, None), slice(-1, -N, -2)]
    integer_choices = list(np.arange(N))
    indices = []
    K = 20
    for _ in range(K):
        array_idx = self.rng.integers(0, 5, size=15)
        curr_idx = self.rng.choice(slice_choices, size=4).tolist()
        _array_idx = self.rng.choice(4)
        curr_idx[_array_idx] = array_idx
        indices.append(tuple(curr_idx))
    for _ in range(K):
        array_idx = self.rng.integers(0, 5, size=15)
        curr_idx = self.rng.choice(integer_choices, size=4).tolist()
        _array_idx = self.rng.choice(4)
        curr_idx[_array_idx] = array_idx
        indices.append(tuple(curr_idx))
    for _ in range(K):
        array_idx = self.rng.integers(0, 5, size=15)
        curr_idx = self.rng.choice(slice_choices, size=4).tolist()
        _array_idx = self.rng.choice(4, size=2, replace=False)
        curr_idx[_array_idx[0]] = array_idx
        curr_idx[_array_idx[1]] = Ellipsis
        indices.append(tuple(curr_idx))
    for _ in range(K):
        array_idx = self.rng.integers(0, 5, size=15)
        curr_idx = self.rng.choice(slice_choices, size=4).tolist()
        _array_idx = self.rng.choice(4)
        bool_arr_shape = self.shape[_array_idx]
        curr_idx[_array_idx] = np.array(self.rng.choice(2, size=bool_arr_shape), dtype=bool)
        indices.append(tuple(curr_idx))
    return indices