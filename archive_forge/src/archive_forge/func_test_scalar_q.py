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
def test_scalar_q(self):
    x = np.arange(12).reshape(3, 4)
    assert_equal(np.percentile(x, 50), 5.5)
    assert_(np.isscalar(np.percentile(x, 50)))
    r0 = np.array([4.0, 5.0, 6.0, 7.0])
    assert_equal(np.percentile(x, 50, axis=0), r0)
    assert_equal(np.percentile(x, 50, axis=0).shape, r0.shape)
    r1 = np.array([1.5, 5.5, 9.5])
    assert_almost_equal(np.percentile(x, 50, axis=1), r1)
    assert_equal(np.percentile(x, 50, axis=1).shape, r1.shape)
    out = np.empty(1)
    assert_equal(np.percentile(x, 50, out=out), 5.5)
    assert_equal(out, 5.5)
    out = np.empty(4)
    assert_equal(np.percentile(x, 50, axis=0, out=out), r0)
    assert_equal(out, r0)
    out = np.empty(3)
    assert_equal(np.percentile(x, 50, axis=1, out=out), r1)
    assert_equal(out, r1)
    x = np.arange(12).reshape(3, 4)
    assert_equal(np.percentile(x, 50, method='lower'), 5.0)
    assert_(np.isscalar(np.percentile(x, 50)))
    r0 = np.array([4.0, 5.0, 6.0, 7.0])
    c0 = np.percentile(x, 50, method='lower', axis=0)
    assert_equal(c0, r0)
    assert_equal(c0.shape, r0.shape)
    r1 = np.array([1.0, 5.0, 9.0])
    c1 = np.percentile(x, 50, method='lower', axis=1)
    assert_almost_equal(c1, r1)
    assert_equal(c1.shape, r1.shape)
    out = np.empty((), dtype=x.dtype)
    c = np.percentile(x, 50, method='lower', out=out)
    assert_equal(c, 5)
    assert_equal(out, 5)
    out = np.empty(4, dtype=x.dtype)
    c = np.percentile(x, 50, method='lower', axis=0, out=out)
    assert_equal(c, r0)
    assert_equal(out, r0)
    out = np.empty(3, dtype=x.dtype)
    c = np.percentile(x, 50, method='lower', axis=1, out=out)
    assert_equal(c, r1)
    assert_equal(out, r1)