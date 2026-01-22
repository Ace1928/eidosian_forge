import os
import sys
import itertools
import traceback
import textwrap
import subprocess
import pytest
import numpy as np
from numpy import array, single, double, csingle, cdouble, dot, identity, matmul
from numpy.core import swapaxes
from numpy import multiply, atleast_2d, inf, asarray
from numpy import linalg
from numpy.linalg import matrix_power, norm, matrix_rank, multi_dot, LinAlgError
from numpy.linalg.linalg import _multi_dot_matrix_chain_order
from numpy.testing import (
def check_qr_stacked(self, a):
    a_type = type(a)
    a_dtype = a.dtype
    m, n = a.shape[-2:]
    k = min(m, n)
    q, r = linalg.qr(a, mode='complete')
    assert_(q.dtype == a_dtype)
    assert_(r.dtype == a_dtype)
    assert_(isinstance(q, a_type))
    assert_(isinstance(r, a_type))
    assert_(q.shape[-2:] == (m, m))
    assert_(r.shape[-2:] == (m, n))
    assert_almost_equal(matmul(q, r), a)
    I_mat = np.identity(q.shape[-1])
    stack_I_mat = np.broadcast_to(I_mat, q.shape[:-2] + (q.shape[-1],) * 2)
    assert_almost_equal(matmul(swapaxes(q, -1, -2).conj(), q), stack_I_mat)
    assert_almost_equal(np.triu(r[..., :, :]), r)
    q1, r1 = linalg.qr(a, mode='reduced')
    assert_(q1.dtype == a_dtype)
    assert_(r1.dtype == a_dtype)
    assert_(isinstance(q1, a_type))
    assert_(isinstance(r1, a_type))
    assert_(q1.shape[-2:] == (m, k))
    assert_(r1.shape[-2:] == (k, n))
    assert_almost_equal(matmul(q1, r1), a)
    I_mat = np.identity(q1.shape[-1])
    stack_I_mat = np.broadcast_to(I_mat, q1.shape[:-2] + (q1.shape[-1],) * 2)
    assert_almost_equal(matmul(swapaxes(q1, -1, -2).conj(), q1), stack_I_mat)
    assert_almost_equal(np.triu(r1[..., :, :]), r1)
    r2 = linalg.qr(a, mode='r')
    assert_(r2.dtype == a_dtype)
    assert_(isinstance(r2, a_type))
    assert_almost_equal(r2, r1)