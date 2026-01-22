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
def test_make_mask(self):
    mask = [0, 1]
    test = make_mask(mask)
    assert_equal(test.dtype, MaskType)
    assert_equal(test, [0, 1])
    mask = np.array([0, 1], dtype=bool)
    test = make_mask(mask)
    assert_equal(test.dtype, MaskType)
    assert_equal(test, [0, 1])
    mdtype = [('a', bool), ('b', bool)]
    mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
    test = make_mask(mask)
    assert_equal(test.dtype, MaskType)
    assert_equal(test, [1, 1])
    mdtype = [('a', bool), ('b', bool)]
    mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
    test = make_mask(mask, dtype=mask.dtype)
    assert_equal(test.dtype, mdtype)
    assert_equal(test, mask)
    mdtype = [('a', float), ('b', float)]
    bdtype = [('a', bool), ('b', bool)]
    mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
    test = make_mask(mask, dtype=mask.dtype)
    assert_equal(test.dtype, bdtype)
    assert_equal(test, np.array([(0, 0), (0, 1)], dtype=bdtype))
    mask = np.array((False, True), dtype='?,?')[()]
    assert_(isinstance(mask, np.void))
    test = make_mask(mask, dtype=mask.dtype)
    assert_equal(test, mask)
    assert_(test is not mask)
    mask = np.array((0, 1), dtype='i4,i4')[()]
    test2 = make_mask(mask, dtype=mask.dtype)
    assert_equal(test2, test)
    bools = [True, False]
    dtypes = [MaskType, float]
    msgformat = 'copy=%s, shrink=%s, dtype=%s'
    for cpy, shr, dt in itertools.product(bools, bools, dtypes):
        res = make_mask(nomask, copy=cpy, shrink=shr, dtype=dt)
        assert_(res is nomask, msgformat % (cpy, shr, dt))