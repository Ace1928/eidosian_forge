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
def test_filled_with_mvoid(self):
    ndtype = [('a', int), ('b', float)]
    a = mvoid((1, 2.0), mask=[(0, 1)], dtype=ndtype)
    test = a.filled()
    assert_equal(tuple(test), (1, default_fill_value(1.0)))
    test = a.filled((-1, -1))
    assert_equal(tuple(test), (1, -1))
    a.fill_value = (-999, -999)
    assert_equal(tuple(a.filled()), (1, -999))