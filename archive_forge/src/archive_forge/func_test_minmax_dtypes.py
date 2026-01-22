import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_minmax_dtypes(self):
    x = np.array([1.0, 1.0, 1.0, -2.0, pi / 2.0, 4.0, 5.0, -10.0, 10.0, 1.0, 2.0, 3.0])
    a10 = 10.0
    an10 = -10.0
    m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    xm = masked_array(x, mask=m1)
    xm.set_fill_value(1e+20)
    float_dtypes = [np.float16, np.float32, np.float64, np.longdouble, np.complex64, np.complex128, np.clongdouble]
    for float_dtype in float_dtypes:
        assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(), float_dtype(a10))
        assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(), float_dtype(an10))
    assert_equal(xm.min(), an10)
    assert_equal(xm.max(), a10)
    for float_dtype in float_dtypes[:4]:
        assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(), float_dtype(a10))
        assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(), float_dtype(an10))
    for float_dtype in float_dtypes[-3:]:
        ym = masked_array([1e+20 + 1j, 1e+20 - 2j, 1e+20 - 1j], mask=[0, 1, 0], dtype=float_dtype)
        assert_equal(ym.min(), float_dtype(1e+20 - 1j))
        assert_equal(ym.max(), float_dtype(1e+20 + 1j))
        zm = masked_array([np.inf + 2j, np.inf + 3j, -np.inf - 1j], mask=[0, 1, 0], dtype=float_dtype)
        assert_equal(zm.min(), float_dtype(-np.inf - 1j))
        assert_equal(zm.max(), float_dtype(np.inf + 2j))
        cmax = np.inf - 1j * np.finfo(np.float64).max
        assert masked_array([-cmax, 0], mask=[0, 1]).max() == -cmax
        assert masked_array([cmax, 0], mask=[0, 1]).min() == cmax