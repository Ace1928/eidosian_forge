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
@pytest.mark.parametrize('chunks', ['auto', (4, 6), (2, 3), (4, 3), (2, 6)])
def test_tensordot_double_contraction_neq2(chunks):
    x = np.arange(24).reshape(4, 6)
    y = da.from_array(x, chunks=chunks)
    assert_eq(da.tensordot(y, y, axes=2), np.tensordot(x, x, axes=2))