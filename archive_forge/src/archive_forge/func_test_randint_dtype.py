from __future__ import annotations
import pytest
import numpy as np
import dask
import dask.array as da
from dask.array.core import Array
from dask.array.utils import assert_eq
from dask.multiprocessing import _dumps, _loads
from dask.utils import key_split
def test_randint_dtype():
    x = da.random.randint(0, 255, size=10, dtype='uint8')
    assert_eq(x, x)
    assert x.dtype == 'uint8'
    assert x.compute().dtype == 'uint8'