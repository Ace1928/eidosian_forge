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
def test_rechunk_with_null_dimensions():
    x = da.from_array(np.ones((24, 24)), chunks=(4, 8))
    assert x.rechunk(chunks=(None, 4)).chunks == da.ones((24, 24), chunks=(4, 4)).chunks
    assert x.rechunk(chunks={0: None, 1: 4}).chunks == da.ones((24, 24), chunks=(4, 4)).chunks