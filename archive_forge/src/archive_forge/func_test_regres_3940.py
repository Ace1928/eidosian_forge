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
@pytest.mark.parametrize('func', [da.cumsum, da.cumprod, da.argmin, da.argmax, da.min, da.max, da.nansum, da.nanmax])
@pytest.mark.parametrize('method', ['sequential', 'blelloch'])
def test_regres_3940(func, method):
    if func in {da.cumsum, da.cumprod}:
        kwargs = {'method': method}
    else:
        kwargs = {}
    a = da.ones((5, 2), chunks=(2, 2))
    assert func(a, **kwargs).name != func(a + 1, **kwargs).name
    assert func(a, axis=0, **kwargs).name != func(a, **kwargs).name
    assert func(a, axis=0, **kwargs).name != func(a, axis=1, **kwargs).name
    if func not in {da.cumsum, da.cumprod, da.argmin, da.argmax}:
        assert func(a, axis=()).name != func(a).name
        assert func(a, axis=()).name != func(a, axis=0).name