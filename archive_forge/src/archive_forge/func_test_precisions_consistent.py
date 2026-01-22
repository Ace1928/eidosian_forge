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
@pytest.mark.xfail(IS_MUSL, reason='gh23049')
@pytest.mark.xfail(IS_WASM, reason="doesn't work")
def test_precisions_consistent(self):
    z = 1 + 1j
    for f in self.funcs:
        fcf = f(np.csingle(z))
        fcd = f(np.cdouble(z))
        fcl = f(np.clongdouble(z))
        assert_almost_equal(fcf, fcd, decimal=6, err_msg='fch-fcd %s' % f)
        assert_almost_equal(fcl, fcd, decimal=15, err_msg='fch-fcl %s' % f)