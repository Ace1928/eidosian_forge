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
def test_eq_on_structured(self):
    ndtype = [('A', int), ('B', int)]
    a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)
    test = a == a
    assert_equal(test.data, [True, True])
    assert_equal(test.mask, [False, False])
    assert_(test.fill_value == True)
    test = a == a[0]
    assert_equal(test.data, [True, False])
    assert_equal(test.mask, [False, False])
    assert_(test.fill_value == True)
    b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
    test = a == b
    assert_equal(test.data, [False, True])
    assert_equal(test.mask, [True, False])
    assert_(test.fill_value == True)
    test = a[0] == b
    assert_equal(test.data, [False, False])
    assert_equal(test.mask, [True, False])
    assert_(test.fill_value == True)
    b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
    test = a == b
    assert_equal(test.data, [True, True])
    assert_equal(test.mask, [False, False])
    assert_(test.fill_value == True)
    ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
    a = array([[(1, (1, 1)), (2, (2, 2))], [(3, (3, 3)), (4, (4, 4))]], mask=[[(0, (1, 0)), (0, (0, 1))], [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
    test = a[0, 0] == a
    assert_equal(test.data, [[True, False], [False, False]])
    assert_equal(test.mask, [[False, False], [False, True]])
    assert_(test.fill_value == True)