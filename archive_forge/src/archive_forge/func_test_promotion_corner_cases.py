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
@np.errstate(all='ignore')
def test_promotion_corner_cases(self):
    for func in self.funcs:
        assert func(np.float16(1)).dtype == np.float16
        assert func(np.uint8(1)).dtype == np.float16
        assert func(np.int16(1)).dtype == np.float32