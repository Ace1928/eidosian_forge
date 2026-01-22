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
def test_complex_nans(self):
    nan = np.nan
    for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
        arg1 = np.array([0, cnan, cnan], dtype=complex)
        arg2 = np.array([cnan, 0, cnan], dtype=complex)
        out = np.array([0, 0, nan], dtype=complex)
        assert_equal(np.fmin(arg1, arg2), out)