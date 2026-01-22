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
def test_arcsincos(self):
    with np.errstate(all='ignore'):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.nan, np.nan]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.arcsin(in_arr), out_arr)
            assert_equal(np.arccos(in_arr), out_arr)
    for callable in [np.arcsin, np.arccos]:
        for value in [np.inf, -np.inf, 2.0, -2.0]:
            for dt in ['e', 'f', 'd']:
                with np.errstate(invalid='raise'):
                    assert_raises(FloatingPointError, callable, np.array(value, dtype=dt))