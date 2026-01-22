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
def test_minmax_blocked(self):
    for dt, sz in [(np.float32, 15), (np.float64, 7)]:
        for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary', max_size=sz):
            for i in range(inp.size):
                inp[:] = np.arange(inp.size, dtype=dt)
                inp[i] = np.nan
                emsg = lambda: '%r\n%s' % (inp, msg)
                with suppress_warnings() as sup:
                    sup.filter(RuntimeWarning, 'invalid value encountered in reduce')
                    assert_(np.isnan(inp.max()), msg=emsg)
                    assert_(np.isnan(inp.min()), msg=emsg)
                inp[i] = 10000000000.0
                assert_equal(inp.max(), 10000000000.0, err_msg=msg)
                inp[i] = -10000000000.0
                assert_equal(inp.min(), -10000000000.0, err_msg=msg)