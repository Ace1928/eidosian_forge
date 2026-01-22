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
@pytest.mark.parametrize('func', ['cumsum', 'cumprod', 'nancumsum', 'nancumprod'])
@pytest.mark.parametrize('use_nan', [False, True])
@pytest.mark.parametrize('axis', [None, 0, 1, -1])
@pytest.mark.parametrize('method', ['sequential', 'blelloch'])
def test_array_cumreduction_axis(func, use_nan, axis, method):
    np_func = getattr(np, func)
    da_func = getattr(da, func)
    s = (10, 11, 12)
    a = np.arange(np.prod(s), dtype=float).reshape(s)
    if use_nan:
        a[1] = np.nan
    d = da.from_array(a, chunks=(4, 5, 6))
    if func in ['cumprod', 'nancumprod'] and method == 'blelloch' and (axis is None):
        with pytest.warns(RuntimeWarning):
            da_func(d, axis=axis, method=method).compute()
            return
    a_r = np_func(a, axis=axis)
    d_r = da_func(d, axis=axis, method=method)
    assert_eq(a_r, d_r)