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
def test_extended_axis(self):
    o = np.random.normal(size=(71, 23))
    x = np.dstack([o] * 10)
    assert_equal(np.median(x, axis=(0, 1)), np.median(o))
    x = np.moveaxis(x, -1, 0)
    assert_equal(np.median(x, axis=(-2, -1)), np.median(o))
    x = x.swapaxes(0, 1).copy()
    assert_equal(np.median(x, axis=(0, -1)), np.median(o))
    assert_equal(np.median(x, axis=(0, 1, 2)), np.median(x, axis=None))
    assert_equal(np.median(x, axis=(0,)), np.median(x, axis=0))
    assert_equal(np.median(x, axis=(-1,)), np.median(x, axis=-1))
    d = np.arange(3 * 5 * 7 * 11).reshape((3, 5, 7, 11))
    np.random.shuffle(d.ravel())
    assert_equal(np.median(d, axis=(0, 1, 2))[0], np.median(d[:, :, :, 0].flatten()))
    assert_equal(np.median(d, axis=(0, 1, 3))[1], np.median(d[:, :, 1, :].flatten()))
    assert_equal(np.median(d, axis=(3, 1, -4))[2], np.median(d[:, :, 2, :].flatten()))
    assert_equal(np.median(d, axis=(3, 1, 2))[2], np.median(d[2, :, :, :].flatten()))
    assert_equal(np.median(d, axis=(3, 2))[2, 1], np.median(d[2, 1, :, :].flatten()))
    assert_equal(np.median(d, axis=(1, -2))[2, 1], np.median(d[2, :, :, 1].flatten()))
    assert_equal(np.median(d, axis=(1, 3))[2, 2], np.median(d[2, :, 2, :].flatten()))