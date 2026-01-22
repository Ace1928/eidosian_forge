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
@pytest.mark.parametrize('dtype', ['d', 'f'])
def test_fp_noncontiguous(self, dtype):
    data = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.2251e-308], dtype=dtype)
    nan = np.array([True, True, False, False, False, False, False, False, False, False])
    inf = np.array([False, False, True, True, False, False, False, False, False, False])
    sign = np.array([False, True, False, True, True, False, True, False, False, True])
    finite = np.array([False, False, False, False, True, True, True, True, True, True])
    out = np.ndarray(data.shape, dtype='bool')
    ncontig_in = data[1::3]
    ncontig_out = out[1::3]
    contig_in = np.array(ncontig_in)
    assert_equal(ncontig_in.flags.c_contiguous, False)
    assert_equal(ncontig_out.flags.c_contiguous, False)
    assert_equal(contig_in.flags.c_contiguous, True)
    assert_equal(np.isnan(ncontig_in, out=ncontig_out), nan[1::3])
    assert_equal(np.isinf(ncontig_in, out=ncontig_out), inf[1::3])
    assert_equal(np.signbit(ncontig_in, out=ncontig_out), sign[1::3])
    assert_equal(np.isfinite(ncontig_in, out=ncontig_out), finite[1::3])
    assert_equal(np.isnan(contig_in, out=ncontig_out), nan[1::3])
    assert_equal(np.isinf(contig_in, out=ncontig_out), inf[1::3])
    assert_equal(np.signbit(contig_in, out=ncontig_out), sign[1::3])
    assert_equal(np.isfinite(contig_in, out=ncontig_out), finite[1::3])
    assert_equal(np.isnan(ncontig_in), nan[1::3])
    assert_equal(np.isinf(ncontig_in), inf[1::3])
    assert_equal(np.signbit(ncontig_in), sign[1::3])
    assert_equal(np.isfinite(ncontig_in), finite[1::3])
    data_split = np.array(np.array_split(data, 2))
    nan_split = np.array(np.array_split(nan, 2))
    inf_split = np.array(np.array_split(inf, 2))
    sign_split = np.array(np.array_split(sign, 2))
    finite_split = np.array(np.array_split(finite, 2))
    assert_equal(np.isnan(data_split), nan_split)
    assert_equal(np.isinf(data_split), inf_split)
    assert_equal(np.signbit(data_split), sign_split)
    assert_equal(np.isfinite(data_split), finite_split)