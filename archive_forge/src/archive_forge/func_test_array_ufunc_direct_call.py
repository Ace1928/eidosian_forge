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
def test_array_ufunc_direct_call(self):
    a = np.array(1)
    with pytest.raises(TypeError):
        a.__array_ufunc__()
    with pytest.raises(TypeError):
        a.__array_ufunc__(1, 2)
    res = a.__array_ufunc__(np.add, '__call__', a, a)
    assert_array_equal(res, a + a)