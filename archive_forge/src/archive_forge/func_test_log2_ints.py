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
@pytest.mark.parametrize('i', range(1, 65))
def test_log2_ints(self, i):
    v = np.log2(2.0 ** i)
    assert_equal(v, float(i), err_msg='at exponent %d' % i)