from __future__ import annotations
import pytest
import numpy as np
import dask.array as da
from dask.array.utils import assert_eq
from dask.array.wrap import ones
def test_singleton_size():
    a = ones(10, dtype='i4', chunks=(4,))
    x = np.array(a)
    assert (x == np.ones(10, dtype='i4')).all()