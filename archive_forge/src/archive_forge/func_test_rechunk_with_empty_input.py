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
def test_rechunk_with_empty_input():
    x = da.ones((24, 24), chunks=(4, 8))
    assert x.rechunk(chunks={}).chunks == x.chunks
    pytest.raises(ValueError, lambda: x.rechunk(chunks=()))