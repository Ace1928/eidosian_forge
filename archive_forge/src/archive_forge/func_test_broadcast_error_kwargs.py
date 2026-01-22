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
def test_broadcast_error_kwargs(self):
    arrs = [np.empty((5, 6, 7))]
    mit = np.broadcast(*arrs)
    mit2 = np.broadcast(*arrs, **{})
    assert_equal(mit.shape, mit2.shape)
    assert_equal(mit.ndim, mit2.ndim)
    assert_equal(mit.nd, mit2.nd)
    assert_equal(mit.numiter, mit2.numiter)
    assert_(mit.iters[0].base is mit2.iters[0].base)
    assert_raises(ValueError, np.broadcast, 1, **{'x': 1})