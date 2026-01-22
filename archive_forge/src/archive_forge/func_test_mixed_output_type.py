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
def test_mixed_output_type():
    y = da.random.default_rng().random((10, 10), chunks=(5, 5))
    y[y < 0.4] = 0
    y = da.ma.masked_equal(y, 0)
    x = da.zeros((10, 1), chunks=(5, 1))
    z = da.concatenate([x, y], axis=1)
    assert z.shape == (10, 11)
    zz = z.compute()
    assert isinstance(zz, np.ma.masked_array)