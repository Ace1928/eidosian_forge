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
def test_rechunk_with_zero_placeholders():
    x = da.ones((24, 24), chunks=((12, 12), (24, 0)))
    y = da.ones((24, 24), chunks=((12, 12), (12, 12)))
    y = y.rechunk(((12, 12), (24, 0)))
    assert x.chunks == y.chunks