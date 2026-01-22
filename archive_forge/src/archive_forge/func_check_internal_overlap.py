import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
def check_internal_overlap(a, manual_expected=None):
    got = internal_overlap(a)
    m = set()
    ranges = tuple((range(n) for n in a.shape))
    for v in itertools.product(*ranges):
        offset = sum((s * w for s, w in zip(a.strides, v)))
        if offset in m:
            expected = True
            break
        else:
            m.add(offset)
    else:
        expected = False
    if got != expected:
        assert_equal(got, expected, err_msg=repr((a.strides, a.shape)))
    if manual_expected is not None and expected != manual_expected:
        assert_equal(expected, manual_expected)
    return got