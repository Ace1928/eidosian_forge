import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple
import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _glibc_older_than
@pytest.mark.parametrize('dtype', ['d', 'f', 'int32', 'int64'])
@pytest.mark.parametrize('big', [True, False])
def test_noncontiguous(self, dtype, big):
    data = np.array([-1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.5, 2.5, -6, 6, -2.2251e-308, -8, 10], dtype=dtype)
    expect = np.array([1.0, -1.0, 0.0, -0.0, -2.2251e-308, 2.5, -2.5, 6, -6, 2.2251e-308, 8, -10], dtype=dtype)
    if big:
        data = np.repeat(data, 10)
        expect = np.repeat(expect, 10)
    out = np.ndarray(data.shape, dtype=dtype)
    ncontig_in = data[1::2]
    ncontig_out = out[1::2]
    contig_in = np.array(ncontig_in)
    assert_array_equal(np.negative(contig_in), expect[1::2])
    assert_array_equal(np.negative(contig_in, out=ncontig_out), expect[1::2])
    assert_array_equal(np.negative(ncontig_in), expect[1::2])
    assert_array_equal(np.negative(ncontig_in, out=ncontig_out), expect[1::2])
    data_split = np.array(np.array_split(data, 2))
    expect_split = np.array(np.array_split(expect, 2))
    assert_equal(np.negative(data_split), expect_split)