from __future__ import annotations
import warnings
from itertools import product
import pytest
import math
import dask
import dask.array as da
from dask.array.rechunk import (
from dask.array.utils import assert_eq
from dask.utils import funcname
def test_rechunk_zero_dim():
    da = pytest.importorskip('dask.array')
    x = da.ones((0, 10, 100), chunks=(0, 10, 10)).rechunk((0, 10, 50))
    assert len(x.compute()) == 0