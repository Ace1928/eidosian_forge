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
def test_domained_binops_d2D(self):
    a = array([[1.0], [2.0], [3.0]], mask=[[False], [True], [True]])
    b = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    test = a / b
    control = array([[1.0 / 2.0, 1.0 / 3.0], [2.0, 2.0], [3.0, 3.0]], mask=[[0, 0], [1, 1], [1, 1]])
    assert_equal(test, control)
    assert_equal(test.data, control.data)
    assert_equal(test.mask, control.mask)
    test = b / a
    control = array([[2.0 / 1.0, 3.0 / 1.0], [4.0, 5.0], [6.0, 7.0]], mask=[[0, 0], [1, 1], [1, 1]])
    assert_equal(test, control)
    assert_equal(test.data, control.data)
    assert_equal(test.mask, control.mask)
    a = array([[1.0], [2.0], [3.0]])
    b = array([[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]], mask=[[0, 0], [0, 0], [0, 1]])
    test = a / b
    control = array([[1.0 / 2, 1.0 / 3], [2.0 / 4, 2.0 / 5], [3.0 / 6, 3]], mask=[[0, 0], [0, 0], [0, 1]])
    assert_equal(test, control)
    assert_equal(test.data, control.data)
    assert_equal(test.mask, control.mask)
    test = b / a
    control = array([[2 / 1.0, 3 / 1.0], [4 / 2.0, 5 / 2.0], [6 / 3.0, 7]], mask=[[0, 0], [0, 0], [0, 1]])
    assert_equal(test, control)
    assert_equal(test.data, control.data)
    assert_equal(test.mask, control.mask)