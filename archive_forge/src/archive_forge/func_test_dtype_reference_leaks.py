import operator
import warnings
import sys
import decimal
from fractions import Fraction
import math
import pytest
import hypothesis
from hypothesis.extra.numpy import arrays
import hypothesis.strategies as st
from functools import partial
import numpy as np
from numpy import ma
from numpy.testing import (
import numpy.lib.function_base as nfb
from numpy.random import rand
from numpy.lib import (
from numpy.core.numeric import normalize_axis_tuple
@pytest.mark.skipif(not HAS_REFCOUNT, reason='Python lacks refcounts')
def test_dtype_reference_leaks(self):
    intp_refcount = sys.getrefcount(np.dtype(np.intp))
    double_refcount = sys.getrefcount(np.dtype(np.double))
    for j in range(10):
        np.bincount([1, 2, 3])
    assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
    assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)
    for j in range(10):
        np.bincount([1, 2, 3], [4, 5, 6])
    assert_equal(sys.getrefcount(np.dtype(np.intp)), intp_refcount)
    assert_equal(sys.getrefcount(np.dtype(np.double)), double_refcount)