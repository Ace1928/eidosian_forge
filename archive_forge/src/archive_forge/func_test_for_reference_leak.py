import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal
import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
from numpy.core._rational_tests import rational
from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_for_reference_leak(self):
    dim = 1
    beg = sys.getrefcount(dim)
    np.zeros([dim] * 10)
    assert_(sys.getrefcount(dim) == beg)
    np.ones([dim] * 10)
    assert_(sys.getrefcount(dim) == beg)
    np.empty([dim] * 10)
    assert_(sys.getrefcount(dim) == beg)
    np.full([dim] * 10, 0)
    assert_(sys.getrefcount(dim) == beg)