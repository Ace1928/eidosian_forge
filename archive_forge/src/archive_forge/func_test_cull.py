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
@pytest.mark.xfail
def test_cull():
    x = da.ones(1000, chunks=(10,))
    for slc in [1, slice(0, 30), slice(0, None, 100)]:
        y = x[slc]
        assert len(y.dask) < len(x.dask)