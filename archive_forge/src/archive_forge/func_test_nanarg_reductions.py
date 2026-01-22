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
@pytest.mark.parametrize(['dfunc', 'func'], [(da.nanargmin, np.nanargmin), (da.nanargmax, np.nanargmax)])
def test_nanarg_reductions(dfunc, func):
    x = np.random.default_rng().random((10, 10, 10))
    x[5] = np.nan
    a = da.from_array(x, chunks=(3, 4, 5))
    assert_eq(dfunc(a), func(x))
    assert_eq(dfunc(a, 0), func(x, 0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        with pytest.raises(ValueError):
            dfunc(a, 1).compute()
        with pytest.raises(ValueError):
            dfunc(a, 2).compute()
        x[:] = np.nan
        a = da.from_array(x, chunks=(3, 4, 5))
        with pytest.raises(ValueError):
            dfunc(a).compute()