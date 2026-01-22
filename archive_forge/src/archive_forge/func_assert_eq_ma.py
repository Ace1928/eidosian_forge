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
def assert_eq_ma(a, b):
    res = a.compute()
    if res is np.ma.masked:
        assert res is b
    else:
        assert type(res) == type(b)
        if hasattr(res, 'mask'):
            np.testing.assert_equal(res.mask, b.mask)
            a = da.ma.filled(a)
            b = np.ma.filled(b)
        assert_eq(a, b, equal_nan=True)