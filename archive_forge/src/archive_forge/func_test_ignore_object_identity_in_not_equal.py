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
def test_ignore_object_identity_in_not_equal(self):
    a = np.array([np.array([1, 2, 3]), None], dtype=object)
    assert_raises(ValueError, np.not_equal, a, a)

    class FunkyType:

        def __ne__(self, other):
            raise TypeError("I won't compare")
    a = np.array([FunkyType()])
    assert_raises(TypeError, np.not_equal, a, a)
    a = np.array([np.nan], dtype=object)
    assert_equal(np.not_equal(a, a), [True])