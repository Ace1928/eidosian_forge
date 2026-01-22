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
@pytest.mark.skipif(sys.platform == 'win32' and sys.maxsize < 2 ** 31 + 1, reason='failures on 32-bit Python, see FIXME below')
@pytest.mark.parametrize('ufunc', UFUNCS_UNARY_FP)
@pytest.mark.parametrize('dtype', ('e', 'f', 'd'))
@pytest.mark.parametrize('data, escape', (([0.03], LTONE_INVALID_ERR), ([0.03] * 32, LTONE_INVALID_ERR), ([-1.0], NEG_INVALID_ERR), ([-1.0] * 32, NEG_INVALID_ERR), ([1.0], ONE_INVALID_ERR), ([1.0] * 32, ONE_INVALID_ERR), ([0.0], BYZERO_ERR), ([0.0] * 32, BYZERO_ERR), ([-0.0], BYZERO_ERR), ([-0.0] * 32, BYZERO_ERR), ([0.5, 0.5, 0.5, np.nan], LTONE_INVALID_ERR), ([0.5, 0.5, 0.5, np.nan] * 32, LTONE_INVALID_ERR), ([np.nan, 1.0, 1.0, 1.0], ONE_INVALID_ERR), ([np.nan, 1.0, 1.0, 1.0] * 32, ONE_INVALID_ERR), ([np.nan], []), ([np.nan] * 32, []), ([0.5, 0.5, 0.5, np.inf], INF_INVALID_ERR + LTONE_INVALID_ERR), ([0.5, 0.5, 0.5, np.inf] * 32, INF_INVALID_ERR + LTONE_INVALID_ERR), ([np.inf, 1.0, 1.0, 1.0], INF_INVALID_ERR), ([np.inf, 1.0, 1.0, 1.0] * 32, INF_INVALID_ERR), ([np.inf], INF_INVALID_ERR), ([np.inf] * 32, INF_INVALID_ERR), ([0.5, 0.5, 0.5, -np.inf], NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR), ([0.5, 0.5, 0.5, -np.inf] * 32, NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR), ([-np.inf, 1.0, 1.0, 1.0], NEG_INVALID_ERR + INF_INVALID_ERR), ([-np.inf, 1.0, 1.0, 1.0] * 32, NEG_INVALID_ERR + INF_INVALID_ERR), ([-np.inf], NEG_INVALID_ERR + INF_INVALID_ERR), ([-np.inf] * 32, NEG_INVALID_ERR + INF_INVALID_ERR)))
def test_unary_spurious_fpexception(self, ufunc, dtype, data, escape):
    if escape and ufunc in escape:
        return
    if ufunc in (np.spacing, np.ceil) and dtype == 'e':
        return
    array = np.array(data, dtype=dtype)
    with assert_no_warnings():
        ufunc(array)