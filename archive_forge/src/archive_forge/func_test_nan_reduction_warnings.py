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
@pytest.mark.parametrize(['dfunc', 'func'], [(da.nanmin, np.nanmin), (da.nanmax, np.nanmax)])
def test_nan_reduction_warnings(dfunc, func):
    x = np.random.default_rng().random((10, 10, 10))
    x[5] = np.nan
    a = da.from_array(x, chunks=(3, 4, 5))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        expected = func(x, 1)
    assert_eq(dfunc(a, 1), expected)