import warnings
import itertools
import sys
import ctypes as ct
import pytest
from pytest import param
import numpy as np
import numpy.core._umath_tests as umt
import numpy.linalg._umath_linalg as uml
import numpy.core._operand_flag_tests as opflag_tests
import numpy.core._rational_tests as _rational_tests
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy.compat import pickle
def test_initial_reduction(self):
    assert_equal(np.maximum.reduce([], initial=0), 0)
    assert_equal(np.minimum.reduce([], initial=np.inf), np.inf)
    assert_equal(np.maximum.reduce([], initial=-np.inf), -np.inf)
    assert_equal(np.minimum.reduce([5], initial=4), 4)
    assert_equal(np.maximum.reduce([4], initial=5), 5)
    assert_equal(np.maximum.reduce([5], initial=4), 5)
    assert_equal(np.minimum.reduce([4], initial=5), 4)
    assert_raises(ValueError, np.minimum.reduce, [], initial=None)
    assert_raises(ValueError, np.add.reduce, [], initial=None)
    with pytest.raises(ValueError):
        np.add.reduce([], initial=None, dtype=object)
    assert_equal(np.add.reduce([], initial=np._NoValue), 0)
    a = np.array([10], dtype=object)
    res = np.add.reduce(a, initial=5)
    assert_equal(res, 15)