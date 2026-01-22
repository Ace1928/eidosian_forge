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
def test_histogramdd_raises_incompat_sample_chunks():
    data = da.random.default_rng().random(size=(10, 3), chunks=(5, 1))
    with pytest.raises(ValueError, match='Input array can only be chunked along the 0th axis'):
        da.histogramdd(data, bins=10, range=((0, 1),) * 3)