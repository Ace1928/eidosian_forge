from __future__ import annotations
import os
import warnings
from contextlib import nullcontext as does_not_warn
from itertools import permutations, zip_longest
import pytest
import itertools
import dask.array as da
import dask.config as config
from dask.array.numpy_compat import NUMPY_GE_122, ComplexWarning
from dask.array.utils import assert_eq, same_keys
from dask.core import get_deps
@pytest.mark.parametrize('func', ['median', 'nanmedian'])
@pytest.mark.parametrize('axis', [0, [0, 2], 1])
def test_median_does_not_rechunk_if_whole_axis_in_one_chunk(axis, func):
    x = np.arange(100).reshape((2, 5, 10))
    d = da.from_array(x, chunks=(2, 1, 10))
    actual = getattr(da, func)(d, axis=axis)
    expected = getattr(np, func)(x, axis=axis)
    assert_eq(actual, expected)
    does_rechunk = 'rechunk' in str(dict(actual.__dask_graph__()))
    if axis == 1:
        assert does_rechunk
    else:
        assert not does_rechunk