import sys
import warnings
import functools
import operator
import pytest
import numpy as np
from numpy.core._multiarray_tests import array_indexing
from itertools import product
from numpy.testing import (
def test_bool_as_int_argument_errors(self):
    a = np.array([[[1]]])
    assert_raises(TypeError, np.reshape, a, (True, -1))
    assert_raises(TypeError, np.reshape, a, (np.bool_(True), -1))
    assert_raises(TypeError, operator.index, np.array(True))
    assert_warns(DeprecationWarning, operator.index, np.True_)
    assert_raises(TypeError, np.take, args=(a, [0], False))