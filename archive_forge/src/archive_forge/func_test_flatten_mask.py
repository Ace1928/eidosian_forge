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
def test_flatten_mask(self):
    mask = np.array([0, 0, 1], dtype=bool)
    assert_equal(flatten_mask(mask), mask)
    mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
    test = flatten_mask(mask)
    control = np.array([0, 0, 0, 1], dtype=bool)
    assert_equal(test, control)
    mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
    data = [(0, (0, 0)), (0, (0, 1))]
    mask = np.array(data, dtype=mdtype)
    test = flatten_mask(mask)
    control = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
    assert_equal(test, control)