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
def test_mvoid_getitem(self):
    ndtype = [('a', int), ('b', int)]
    a = masked_array([(1, 2), (3, 4)], mask=[(0, 0), (1, 0)], dtype=ndtype)
    f = a[0]
    assert_(isinstance(f, mvoid))
    assert_equal((f[0], f['a']), (1, 1))
    assert_equal(f['b'], 2)
    f = a[1]
    assert_(isinstance(f, mvoid))
    assert_(f[0] is masked)
    assert_(f['a'] is masked)
    assert_equal(f[1], 4)
    A = masked_array(data=[([0, 1],)], mask=[([True, False],)], dtype=[('A', '>i2', (2,))])
    assert_equal(A[0]['A'], A['A'][0])
    assert_equal(A[0]['A'], masked_array(data=[0, 1], mask=[True, False], dtype='>i2'))