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
def test_topython(self):
    assert_equal(1, int(array(1)))
    assert_equal(1.0, float(array(1)))
    assert_equal(1, int(array([[[1]]])))
    assert_equal(1.0, float(array([[1]])))
    assert_raises(TypeError, float, array([1, 1]))
    with suppress_warnings() as sup:
        sup.filter(UserWarning, 'Warning: converting a masked element')
        assert_(np.isnan(float(array([1], mask=[1]))))
        a = array([1, 2, 3], mask=[1, 0, 0])
        assert_raises(TypeError, lambda: float(a))
        assert_equal(float(a[-1]), 3.0)
        assert_(np.isnan(float(a[0])))
    assert_raises(TypeError, int, a)
    assert_equal(int(a[-1]), 3)
    assert_raises(MAError, lambda: int(a[0]))