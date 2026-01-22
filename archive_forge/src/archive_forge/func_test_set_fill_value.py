from __future__ import annotations
import random
import sys
from copy import deepcopy
from itertools import product
import numpy as np
import pytest
import dask.array as da
from dask.array.numpy_compat import NUMPY_GE_123, ComplexWarning
from dask.array.utils import assert_eq
from dask.base import tokenize
from dask.utils import typename
def test_set_fill_value():
    x = np.random.default_rng().integers(0, 10, (10, 10))
    dx = da.from_array(x, chunks=(3, 4))
    mx = np.ma.masked_greater(x, 3)
    dmx = da.ma.masked_greater(dx, 3)
    da.ma.set_fill_value(dmx, -10)
    np.ma.set_fill_value(mx, -10)
    assert_eq_ma(dmx, mx)
    da.ma.set_fill_value(dx, -10)
    np.ma.set_fill_value(x, -10)
    assert_eq_ma(dx, x)
    with pytest.raises(TypeError):
        da.ma.set_fill_value(dmx, 1e+20)
    with pytest.raises(ValueError):
        da.ma.set_fill_value(dmx, dx)