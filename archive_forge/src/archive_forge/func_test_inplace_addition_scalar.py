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
def test_inplace_addition_scalar(self):
    x, y, xm = self.intdata
    xm[2] = masked
    x += 1
    assert_equal(x, y + 1)
    xm += 1
    assert_equal(xm, y + 1)
    x, _, xm = self.floatdata
    id1 = x.data.ctypes.data
    x += 1.0
    assert_(id1 == x.data.ctypes.data)
    assert_equal(x, y + 1.0)