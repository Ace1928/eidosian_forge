import sys
import warnings
import copy
import operator
import itertools
import textwrap
import pytest
from functools import reduce
import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
from numpy.ma.core import (
from numpy.compat import pickle
def test_fillvalue_exotic_dtype(self):
    _check_fill_value = np.ma.core._check_fill_value
    ndtype = [('i', int), ('s', '|S8'), ('f', float)]
    control = np.array((default_fill_value(0), default_fill_value('0'), default_fill_value(0.0)), dtype=ndtype)
    assert_equal(_check_fill_value(None, ndtype), control)
    ndtype = [('f0', float, (2, 2))]
    control = np.array((default_fill_value(0.0),), dtype=[('f0', float)]).astype(ndtype)
    assert_equal(_check_fill_value(None, ndtype), control)
    control = np.array((0,), dtype=[('f0', float)]).astype(ndtype)
    assert_equal(_check_fill_value(0, ndtype), control)
    ndtype = np.dtype('int, (2,3)float, float')
    control = np.array((default_fill_value(0), default_fill_value(0.0), default_fill_value(0.0)), dtype='int, float, float').astype(ndtype)
    test = _check_fill_value(None, ndtype)
    assert_equal(test, control)
    control = np.array((0, 0, 0), dtype='int, float, float').astype(ndtype)
    assert_equal(_check_fill_value(0, ndtype), control)
    M = masked_array(control)
    assert_equal(M['f1'].fill_value.ndim, 0)