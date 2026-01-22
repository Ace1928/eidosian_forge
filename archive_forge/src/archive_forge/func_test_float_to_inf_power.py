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
def test_float_to_inf_power(self):
    for dt in [np.float32, np.float64]:
        a = np.array([1, 1, 2, 2, -2, -2, np.inf, -np.inf], dt)
        b = np.array([np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf, -np.inf], dt)
        r = np.array([1, 1, np.inf, 0, np.inf, 0, np.inf, 0], dt)
        assert_equal(np.power(a, b), r)