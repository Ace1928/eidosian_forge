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
def test_lcm_overflow(self):
    big = np.int32(np.iinfo(np.int32).max // 11)
    a = 2 * big
    b = 5 * big
    assert_equal(np.lcm(a, b), 10 * big)