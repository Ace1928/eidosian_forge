import itertools
import sys
import platform
import pytest
import numpy as np
from numpy.testing import (
def test_einsum_broadcast(self):
    A = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    B = np.arange(3)
    ref = np.einsum('ijk,j->ijk', A, B, optimize=False)
    for opt in [True, False]:
        assert_equal(np.einsum('ij...,j...->ij...', A, B, optimize=opt), ref)
        assert_equal(np.einsum('ij...,...j->ij...', A, B, optimize=opt), ref)
        assert_equal(np.einsum('ij...,j->ij...', A, B, optimize=opt), ref)
    A = np.arange(12).reshape((4, 3))
    B = np.arange(6).reshape((3, 2))
    ref = np.einsum('ik,kj->ij', A, B, optimize=False)
    for opt in [True, False]:
        assert_equal(np.einsum('ik...,k...->i...', A, B, optimize=opt), ref)
        assert_equal(np.einsum('ik...,...kj->i...j', A, B, optimize=opt), ref)
        assert_equal(np.einsum('...k,kj', A, B, optimize=opt), ref)
        assert_equal(np.einsum('ik,k...->i...', A, B, optimize=opt), ref)
    dims = [2, 3, 4, 5]
    a = np.arange(np.prod(dims)).reshape(dims)
    v = np.arange(dims[2])
    ref = np.einsum('ijkl,k->ijl', a, v, optimize=False)
    for opt in [True, False]:
        assert_equal(np.einsum('ijkl,k', a, v, optimize=opt), ref)
        assert_equal(np.einsum('...kl,k', a, v, optimize=opt), ref)
        assert_equal(np.einsum('...kl,k...', a, v, optimize=opt), ref)
    J, K, M = (160, 160, 120)
    A = np.arange(J * K * M).reshape(1, 1, 1, J, K, M)
    B = np.arange(J * K * M * 3).reshape(J, K, M, 3)
    ref = np.einsum('...lmn,...lmno->...o', A, B, optimize=False)
    for opt in [True, False]:
        assert_equal(np.einsum('...lmn,lmno->...o', A, B, optimize=opt), ref)