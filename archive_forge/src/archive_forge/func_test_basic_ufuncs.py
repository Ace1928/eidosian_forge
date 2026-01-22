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
def test_basic_ufuncs(self):
    x, y, a10, m1, m2, xm, ym, z, zm, xf = self.d
    assert_equal(np.cos(x), cos(xm))
    assert_equal(np.cosh(x), cosh(xm))
    assert_equal(np.sin(x), sin(xm))
    assert_equal(np.sinh(x), sinh(xm))
    assert_equal(np.tan(x), tan(xm))
    assert_equal(np.tanh(x), tanh(xm))
    assert_equal(np.sqrt(abs(x)), sqrt(xm))
    assert_equal(np.log(abs(x)), log(xm))
    assert_equal(np.log10(abs(x)), log10(xm))
    assert_equal(np.exp(x), exp(xm))
    assert_equal(np.arcsin(z), arcsin(zm))
    assert_equal(np.arccos(z), arccos(zm))
    assert_equal(np.arctan(z), arctan(zm))
    assert_equal(np.arctan2(x, y), arctan2(xm, ym))
    assert_equal(np.absolute(x), absolute(xm))
    assert_equal(np.angle(x + 1j * y), angle(xm + 1j * ym))
    assert_equal(np.angle(x + 1j * y, deg=True), angle(xm + 1j * ym, deg=True))
    assert_equal(np.equal(x, y), equal(xm, ym))
    assert_equal(np.not_equal(x, y), not_equal(xm, ym))
    assert_equal(np.less(x, y), less(xm, ym))
    assert_equal(np.greater(x, y), greater(xm, ym))
    assert_equal(np.less_equal(x, y), less_equal(xm, ym))
    assert_equal(np.greater_equal(x, y), greater_equal(xm, ym))
    assert_equal(np.conjugate(x), conjugate(xm))