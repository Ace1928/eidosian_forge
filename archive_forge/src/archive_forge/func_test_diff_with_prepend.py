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
def test_diff_with_prepend(self):
    x = np.array([1, 2, 2, 3, 4, 2, 1, 1])
    a = np.ma.masked_equal(x[3:], value=2)
    a_prep = np.ma.masked_equal(x[:3], value=2)
    diff1 = np.ma.diff(a, prepend=a_prep, axis=0)
    b = np.ma.masked_equal(x, value=2)
    diff2 = np.ma.diff(b, axis=0)
    assert_(np.ma.allequal(diff1, diff2))