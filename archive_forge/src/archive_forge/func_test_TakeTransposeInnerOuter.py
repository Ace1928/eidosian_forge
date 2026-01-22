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
def test_TakeTransposeInnerOuter(self):
    x = arange(24)
    y = np.arange(24)
    x[5:6] = masked
    x = x.reshape(2, 3, 4)
    y = y.reshape(2, 3, 4)
    assert_equal(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1)))
    assert_equal(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1))
    assert_equal(np.inner(filled(x, 0), filled(y, 0)), inner(x, y))
    assert_equal(np.outer(filled(x, 0), filled(y, 0)), outer(x, y))
    y = array(['abc', 1, 'def', 2, 3], object)
    y[2] = masked
    t = take(y, [0, 3, 4])
    assert_(t[0] == 'abc')
    assert_(t[1] == 2)
    assert_(t[2] == 3)