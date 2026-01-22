from __future__ import annotations
import itertools
import warnings
import pytest
from tlz import merge
import dask
import dask.array as da
from dask import config
from dask.array.chunk import getitem
from dask.array.slicing import (
from dask.array.utils import assert_eq, same_keys
@pytest.mark.slow
def test_slicing_none_int_ellipes():
    shape = (2, 3, 5, 7, 11)
    x = np.arange(np.prod(shape)).reshape(shape)
    y = da.core.asarray(x)
    for ind in itertools.product(indexers, indexers, indexers, indexers):
        if ind.count(Ellipsis) > 1:
            continue
        assert_eq(x[ind], y[ind])