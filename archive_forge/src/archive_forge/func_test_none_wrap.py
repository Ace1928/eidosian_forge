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
def test_none_wrap(self):

    class A:

        def __array__(self):
            return np.zeros(1)

        def __array_wrap__(self, arr, context=None):
            return None
    a = A()
    assert_equal(ncu.maximum(a, a), None)