import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_lange(self):
    a = np.array([[-149, -50, -154], [537, 180, 546], [-27, -9, -25]])
    for dtype in 'fdFD':
        for norm_str in 'Mm1OoIiFfEe':
            a1 = a.astype(dtype)
            if dtype.isupper():
                a1[0, 0] += 1j
            lange, = get_lapack_funcs(('lange',), (a1,))
            value = lange(norm_str, a1)
            if norm_str in 'FfEe':
                if dtype in 'Ff':
                    decimal = 3
                else:
                    decimal = 7
                ref = np.sqrt(np.sum(np.square(np.abs(a1))))
                assert_almost_equal(value, ref, decimal)
            else:
                if norm_str in 'Mm':
                    ref = np.max(np.abs(a1))
                elif norm_str in '1Oo':
                    ref = np.max(np.sum(np.abs(a1), axis=0))
                elif norm_str in 'Ii':
                    ref = np.max(np.sum(np.abs(a1), axis=1))
                assert_equal(value, ref)