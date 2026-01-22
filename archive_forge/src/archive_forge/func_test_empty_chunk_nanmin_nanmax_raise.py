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
@pytest.mark.parametrize('func', ['nanmin', 'nanmax'])
def test_empty_chunk_nanmin_nanmax_raise(func):
    x = np.arange(10).reshape(2, 5)
    d = da.from_array(x, chunks=2)
    d = d[d > 9]
    x = x[x > 9]
    d = d.compute_chunk_sizes()
    with pytest.raises(ValueError) as err_np:
        getattr(np, func)(x)
    with pytest.raises(ValueError) as err_da:
        d = getattr(da, func)(d)
        d.compute()
    assert str(err_np.value) == str(err_da.value)