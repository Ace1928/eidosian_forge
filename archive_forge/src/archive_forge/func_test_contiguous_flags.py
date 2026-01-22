import sys
import numpy as np
from numpy.core._rational_tests import rational
import pytest
from numpy.testing import (
def test_contiguous_flags():
    a = np.ones((4, 4, 1))[::2, :, :]
    a.strides = a.strides[:2] + (-123,)
    b = np.ones((2, 2, 1, 2, 2)).swapaxes(3, 4)

    def check_contig(a, ccontig, fcontig):
        assert_(a.flags.c_contiguous == ccontig)
        assert_(a.flags.f_contiguous == fcontig)
    check_contig(a, False, False)
    check_contig(b, False, False)
    check_contig(np.empty((2, 2, 0, 2, 2)), True, True)
    check_contig(np.array([[[1], [2]]], order='F'), True, True)
    check_contig(np.empty((2, 2)), True, False)
    check_contig(np.empty((2, 2), order='F'), False, True)
    check_contig(np.array(a, copy=False), False, False)
    check_contig(np.array(a, copy=False, order='C'), True, False)
    check_contig(np.array(a, ndmin=4, copy=False, order='F'), False, True)
    check_contig(a[0], True, True)
    check_contig(a[None, ::4, ..., None], True, True)
    check_contig(b[0, 0, ...], False, True)
    check_contig(b[:, :, 0:0, :, :], True, True)
    check_contig(a.ravel(), True, True)
    check_contig(np.ones((1, 3, 1)).squeeze(), True, True)