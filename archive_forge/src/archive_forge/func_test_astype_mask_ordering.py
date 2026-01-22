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
def test_astype_mask_ordering():
    descr = np.dtype([('v', int, 3), ('x', [('y', float)])])
    x = array([[([1, 2, 3], (1.0,)), ([1, 2, 3], (2.0,))], [([1, 2, 3], (3.0,)), ([1, 2, 3], (4.0,))]], dtype=descr)
    x[0]['v'][0] = np.ma.masked
    x_a = x.astype(descr)
    assert x_a.dtype.names == np.dtype(descr).names
    assert x_a.mask.dtype.names == np.dtype(descr).names
    assert_equal(x, x_a)
    assert_(x is x.astype(x.dtype, copy=False))
    assert_equal(type(x.astype(x.dtype, subok=False)), np.ndarray)
    x_f = x.astype(x.dtype, order='F')
    assert_(x_f.flags.f_contiguous)
    assert_(x_f.mask.flags.f_contiguous)
    x_a2 = np.array(x, dtype=descr, subok=True)
    assert x_a2.dtype.names == np.dtype(descr).names
    assert x_a2.mask.dtype.names == np.dtype(descr).names
    assert_equal(x, x_a2)
    assert_(x is np.array(x, dtype=descr, copy=False, subok=True))
    x_f2 = np.array(x, dtype=x.dtype, order='F', subok=True)
    assert_(x_f2.flags.f_contiguous)
    assert_(x_f2.mask.flags.f_contiguous)