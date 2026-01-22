import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffering_delayed_alloc():
    a = np.arange(6)
    b = np.arange(1, dtype='f4')
    i = nditer([a, b], ['buffered', 'delay_bufalloc', 'multi_index', 'reduce_ok'], ['readwrite'], casting='unsafe', op_dtypes='f4')
    assert_(i.has_delayed_bufalloc)
    assert_raises(ValueError, lambda i: i.multi_index, i)
    assert_raises(ValueError, lambda i: i[0], i)
    assert_raises(ValueError, lambda i: i[0:2], i)

    def assign_iter(i):
        i[0] = 0
    assert_raises(ValueError, assign_iter, i)
    i.reset()
    assert_(not i.has_delayed_bufalloc)
    assert_equal(i.multi_index, (0,))
    with i:
        assert_equal(i[0], 0)
        i[1] = 1
        assert_equal(i[0:2], [0, 1])
        assert_equal([[x[0][()], x[1][()]] for x in i], list(zip(range(6), [1] * 6)))