import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
@pytest.mark.slow
def test_unary_gufunc_fuzz(self):
    shapes = [7, 13, 8, 21, 29, 32]
    gufunc = _umath_tests.euclidean_pdist
    rng = np.random.RandomState(1234)
    for ndim in range(2, 6):
        x = rng.rand(*shapes[:ndim])
        it = iter_random_view_pairs(x, same_steps=False, equal_size=True)
        min_count = 500 // (ndim + 1) ** 2
        overlapping = 0
        while overlapping < min_count:
            a, b = next(it)
            if min(a.shape[-2:]) < 2 or min(b.shape[-2:]) < 2 or a.shape[-1] < 2:
                continue
            if b.shape[-1] > b.shape[-2]:
                b = b[..., 0, :]
            else:
                b = b[..., :, 0]
            n = a.shape[-2]
            p = n * (n - 1) // 2
            if p <= b.shape[-1] and p > 0:
                b = b[..., :p]
            else:
                n = max(2, int(np.sqrt(b.shape[-1])) // 2)
                p = n * (n - 1) // 2
                a = a[..., :n, :]
                b = b[..., :p]
            if np.shares_memory(a, b):
                overlapping += 1
            with np.errstate(over='ignore', invalid='ignore'):
                assert_copy_equivalent(gufunc, [a], out=b)