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
def test_failing_out_wrap(self):
    singleton = np.array([1.0])

    class Ok(np.ndarray):

        def __array_wrap__(self, obj):
            return singleton

    class Bad(np.ndarray):

        def __array_wrap__(self, obj):
            raise RuntimeError
    ok = np.empty(1).view(Ok)
    bad = np.empty(1).view(Bad)
    for i in range(10):
        assert_raises(RuntimeError, ncu.frexp, 1, ok, bad)