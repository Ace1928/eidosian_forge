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
def test_select_multidimension():
    x = np.random.default_rng().random((100, 50, 2))
    y = da.from_array(x, chunks=(50, 50, 1))
    res_x = np.select([x < 0, x > 2, x > 1], [x, x * 2, x * 3], default=1)
    res_y = da.select([y < 0, y > 2, y > 1], [y, y * 2, y * 3], default=1)
    assert isinstance(res_y, da.Array)
    assert_eq(res_y, res_x)