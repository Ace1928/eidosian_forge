import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
@pytest.mark.parametrize('dtype,ex_val', itertools.product(np.sctypes['int'] + np.sctypes['uint'], ('np.array([fo.max, 1, 2, 1, 1, 2, 3], dtype=dtype)', 'np.array([fo.min, 1, -2, 1, 1, 2, -3]).astype(dtype)', 'np.arange(fo.min, fo.min+(100*10), 10, dtype=dtype)', 'np.array(range(fo.max-(100*7), fo.max, 7)).astype(dtype)')))
def test_division_int_reduce(self, dtype, ex_val):
    fo = np.iinfo(dtype)
    a = eval(ex_val)
    lst = a.tolist()
    c_div = lambda n, d: 0 if d == 0 or (n and n == fo.min and (d == -1)) else n // d
    with np.errstate(divide='ignore'):
        div_a = np.floor_divide.reduce(a)
    div_lst = reduce(c_div, lst)
    msg = 'Reduce floor integer division check'
    assert div_a == div_lst, msg
    with np.errstate(divide='raise', over='raise'):
        with pytest.raises(FloatingPointError, match='divide by zero encountered in reduce'):
            np.floor_divide.reduce(np.arange(-100, 100).astype(dtype))
        if fo.min:
            with pytest.raises(FloatingPointError, match='overflow encountered in reduce'):
                np.floor_divide.reduce(np.array([fo.min, 1, -1], dtype=dtype))