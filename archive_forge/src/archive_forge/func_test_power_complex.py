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
def test_power_complex(self):
    x = np.array([1 + 2j, 2 + 3j, 3 + 4j])
    assert_equal(x ** 0, [1.0, 1.0, 1.0])
    assert_equal(x ** 1, x)
    assert_almost_equal(x ** 2, [-3 + 4j, -5 + 12j, -7 + 24j])
    assert_almost_equal(x ** 3, [(1 + 2j) ** 3, (2 + 3j) ** 3, (3 + 4j) ** 3])
    assert_almost_equal(x ** 4, [(1 + 2j) ** 4, (2 + 3j) ** 4, (3 + 4j) ** 4])
    assert_almost_equal(x ** (-1), [1 / (1 + 2j), 1 / (2 + 3j), 1 / (3 + 4j)])
    assert_almost_equal(x ** (-2), [1 / (1 + 2j) ** 2, 1 / (2 + 3j) ** 2, 1 / (3 + 4j) ** 2])
    assert_almost_equal(x ** (-3), [(-11 + 2j) / 125, (-46 - 9j) / 2197, (-117 - 44j) / 15625])
    assert_almost_equal(x ** 0.5, [ncu.sqrt(1 + 2j), ncu.sqrt(2 + 3j), ncu.sqrt(3 + 4j)])
    norm = 1.0 / (x ** 14)[0]
    assert_almost_equal(x ** 14 * norm, [i * norm for i in [-76443 + 16124j, 23161315 + 58317492j, 5583548873 + 2465133864j]])

    def assert_complex_equal(x, y):
        assert_array_equal(x.real, y.real)
        assert_array_equal(x.imag, y.imag)
    for z in [complex(0, np.inf), complex(1, np.inf)]:
        z = np.array([z], dtype=np.complex_)
        with np.errstate(invalid='ignore'):
            assert_complex_equal(z ** 1, z)
            assert_complex_equal(z ** 2, z * z)
            assert_complex_equal(z ** 3, z * z * z)