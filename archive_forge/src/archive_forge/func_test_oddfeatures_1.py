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
def test_oddfeatures_1(self):
    x = arange(20)
    x = x.reshape(4, 5)
    x.flat[5] = 12
    assert_(x[1, 0] == 12)
    z = x + 10j * x
    assert_equal(z.real, x)
    assert_equal(z.imag, 10 * x)
    assert_equal((z * conjugate(z)).real, 101 * x * x)
    z.imag[...] = 0.0
    x = arange(10)
    x[3] = masked
    assert_(str(x[3]) == str(masked))
    c = x >= 8
    assert_(count(where(c, masked, masked)) == 0)
    assert_(shape(where(c, masked, masked)) == c.shape)
    z = masked_where(c, x)
    assert_(z.dtype is x.dtype)
    assert_(z[3] is masked)
    assert_(z[4] is not masked)
    assert_(z[7] is not masked)
    assert_(z[8] is masked)
    assert_(z[9] is masked)
    assert_equal(x, z)