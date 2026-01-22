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
def test_ufunc_override_exception(self):

    class A:

        def __array_ufunc__(self, *a, **kwargs):
            raise ValueError('oops')
    a = A()
    assert_raises(ValueError, np.negative, 1, out=a)
    assert_raises(ValueError, np.negative, a)
    assert_raises(ValueError, np.divide, 1.0, a)