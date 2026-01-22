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
def test_put_nomask(self):
    x = zeros(10)
    z = array([3.0, -1.0], mask=[False, True])
    x.put([1, 2], z)
    assert_(x[0] is not masked)
    assert_equal(x[0], 0)
    assert_(x[1] is not masked)
    assert_equal(x[1], 3)
    assert_(x[2] is masked)
    assert_(x[3] is not masked)
    assert_equal(x[3], 0)