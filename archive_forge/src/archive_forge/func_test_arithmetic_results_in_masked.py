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
def test_arithmetic_results_in_masked():
    mask = np.array([[True, False], [True, True], [False, True]])
    x = np.arange(6).reshape((3, 2))
    masked = np.ma.array(x, mask=mask)
    dx = da.from_array(x, chunks=(2, 2))
    res = dx + masked
    sol = x + masked
    assert_eq(res, sol)
    assert isinstance(res.compute(), np.ma.masked_array)