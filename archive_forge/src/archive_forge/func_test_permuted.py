import sys
import hashlib
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
from numpy.random import Generator, MT19937, SeedSequence, RandomState
@pytest.mark.parametrize('dtype', [int, object])
@pytest.mark.parametrize('axis, expected', [(None, np.array([[3, 7, 0, 9, 10, 11], [8, 4, 2, 5, 1, 6]])), (0, np.array([[6, 1, 2, 9, 10, 11], [0, 7, 8, 3, 4, 5]])), (1, np.array([[5, 3, 4, 0, 2, 1], [11, 9, 10, 6, 8, 7]]))])
def test_permuted(self, dtype, axis, expected):
    random = Generator(MT19937(self.seed))
    x = np.arange(12).reshape(2, 6).astype(dtype)
    random.permuted(x, axis=axis, out=x)
    assert_array_equal(x, expected)
    random = Generator(MT19937(self.seed))
    x = np.arange(12).reshape(2, 6).astype(dtype)
    y = random.permuted(x, axis=axis)
    assert y.dtype == dtype
    assert_array_equal(y, expected)