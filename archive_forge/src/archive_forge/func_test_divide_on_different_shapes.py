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
def test_divide_on_different_shapes(self):
    x = arange(6, dtype=float)
    x.shape = (2, 3)
    y = arange(3, dtype=float)
    z = x / y
    assert_equal(z, [[-1.0, 1.0, 1.0], [-1.0, 4.0, 2.5]])
    assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])
    z = x / y[None, :]
    assert_equal(z, [[-1.0, 1.0, 1.0], [-1.0, 4.0, 2.5]])
    assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])
    y = arange(2, dtype=float)
    z = x / y[:, None]
    assert_equal(z, [[-1.0, -1.0, -1.0], [3.0, 4.0, 5.0]])
    assert_equal(z.mask, [[1, 1, 1], [0, 0, 0]])