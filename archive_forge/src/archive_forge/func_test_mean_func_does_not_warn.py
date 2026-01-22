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
def test_mean_func_does_not_warn():
    xr = pytest.importorskip('xarray')
    a = xr.DataArray(da.from_array(np.full((10, 10), np.nan)))
    with warnings.catch_warnings(record=True) as rec:
        a.mean().compute()
    assert not rec