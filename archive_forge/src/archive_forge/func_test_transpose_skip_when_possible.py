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
def test_transpose_skip_when_possible():
    x = da.ones((2, 3, 4), chunks=3)
    assert x.transpose((0, 1, 2)) is x
    assert x.transpose((-3, -2, -1)) is x