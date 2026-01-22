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
def test_cumsumprod(self):
    x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX = self.d
    mXcp = mX.cumsum(0)
    assert_equal(mXcp._data, mX.filled(0).cumsum(0))
    mXcp = mX.cumsum(1)
    assert_equal(mXcp._data, mX.filled(0).cumsum(1))
    mXcp = mX.cumprod(0)
    assert_equal(mXcp._data, mX.filled(1).cumprod(0))
    mXcp = mX.cumprod(1)
    assert_equal(mXcp._data, mX.filled(1).cumprod(1))