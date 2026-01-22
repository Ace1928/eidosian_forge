import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_reduction():
    a = np.arange(6)
    i = nditer([a, None], ['reduce_ok'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0], [-1]])
    with i:
        i.operands[1][...] = 0
        for x, y in i:
            y[...] += x
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))
    a = np.arange(6).reshape(2, 3)
    i = nditer([a, None], ['reduce_ok', 'external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[[0, 1], [-1, -1]])
    with i:
        i.operands[1][...] = 0
        assert_equal(i[1].shape, (6,))
        assert_equal(i[1].strides, (0,))
        for x, y in i:
            for j in range(len(y)):
                y[j] += x[j]
        assert_equal(i.operands[1].ndim, 0)
        assert_equal(i.operands[1], np.sum(a))
    a = np.ones((2, 3, 5))
    it1 = nditer([a, None], ['reduce_ok', 'external_loop'], [['readonly'], ['readwrite', 'allocate']], op_axes=[None, [0, -1, 1]])
    it2 = nditer([a, None], ['reduce_ok', 'external_loop', 'buffered', 'delay_bufalloc'], [['readonly'], ['readwrite', 'allocate']], op_axes=[None, [0, -1, 1]], buffersize=10)
    with it1, it2:
        it1.operands[1].fill(0)
        it2.operands[1].fill(0)
        it2.reset()
        for x in it1:
            x[1][...] += x[0]
        for x in it2:
            x[1][...] += x[0]
        assert_equal(it1.operands[1], it2.operands[1])
        assert_equal(it2.operands[1].sum(), a.size)