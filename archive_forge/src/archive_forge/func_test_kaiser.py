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
def test_kaiser(self, dtype: str, M: int) -> None:
    scalar = np.array(M, dtype=dtype)[()]
    w = kaiser(scalar, 0)
    if dtype == 'O':
        ref_dtype = np.float64
    else:
        ref_dtype = np.result_type(scalar.dtype, np.float64)
    assert w.dtype == ref_dtype
    assert_equal(w, flipud(w))
    if scalar < 1:
        assert_array_equal(w, np.array([]))
    elif scalar == 1:
        assert_array_equal(w, np.ones(1))
    else:
        assert_almost_equal(np.sum(w, axis=0), 10, 15)