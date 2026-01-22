import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_no_inner_dim_coalescing():
    a = arange(24).reshape(2, 3, 4)[:, :, :-1]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (3,))
    a = arange(24).reshape(2, 3, 4)[:, :-1, :]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 2)
    assert_equal(i[0].shape, (8,))
    a = arange(24).reshape(2, 3, 4)[:-1, :, :]
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (12,))
    a = arange(24).reshape(1, 1, 2, 1, 1, 3, 1, 1, 4, 1, 1)
    i = nditer(a, ['external_loop'], [['readonly']])
    assert_equal(i.ndim, 1)
    assert_equal(i[0].shape, (24,))