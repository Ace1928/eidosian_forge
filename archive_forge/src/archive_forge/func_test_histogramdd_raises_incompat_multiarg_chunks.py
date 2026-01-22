from __future__ import annotations
import contextlib
import itertools
import pickle
import sys
import warnings
from numbers import Number
import pytest
import dask
from dask.delayed import delayed
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, NUMPY_GE_200, AxisError
from dask.array.utils import assert_eq, same_keys
def test_histogramdd_raises_incompat_multiarg_chunks():
    rng = da.random.default_rng()
    x = rng.random(size=(10,), chunks=2)
    y = rng.random(size=(10,), chunks=2)
    z = rng.random(size=(10,), chunks=5)
    with pytest.raises(ValueError, match='All coordinate arrays must be chunked identically.'):
        da.histogramdd((x, y, z), bins=(3,) * 3, range=((0, 1),) * 3)