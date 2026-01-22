import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def test_internal_overlap_fuzz():
    x = np.arange(1).astype(np.int8)
    overlap = 0
    no_overlap = 0
    min_count = 100
    rng = np.random.RandomState(1234)
    while min(overlap, no_overlap) < min_count:
        ndim = rng.randint(1, 4, dtype=np.intp)
        strides = tuple((rng.randint(-1000, 1000, dtype=np.intp) for j in range(ndim)))
        shape = tuple((rng.randint(1, 30, dtype=np.intp) for j in range(ndim)))
        a = as_strided(x, strides=strides, shape=shape)
        result = check_internal_overlap(a)
        if result:
            overlap += 1
        else:
            no_overlap += 1