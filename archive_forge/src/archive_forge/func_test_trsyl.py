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
def test_trsyl(self):
    a = np.array([[1, 2], [0, 4]])
    b = np.array([[5, 6], [0, 8]])
    c = np.array([[9, 10], [11, 12]])
    trans = 'T'
    for dtype in 'fdFD':
        a1, b1, c1 = (a.astype(dtype), b.astype(dtype), c.astype(dtype))
        trsyl, = get_lapack_funcs(('trsyl',), (a1,))
        if dtype.isupper():
            a1[0] += 1j
            trans = 'C'
        x, scale, info = trsyl(a1, b1, c1)
        assert_array_almost_equal(np.dot(a1, x) + np.dot(x, b1), scale * c1)
        x, scale, info = trsyl(a1, b1, c1, trana=trans, tranb=trans)
        assert_array_almost_equal(np.dot(a1.conjugate().T, x) + np.dot(x, b1.conjugate().T), scale * c1, decimal=4)
        x, scale, info = trsyl(a1, b1, c1, isgn=-1)
        assert_array_almost_equal(np.dot(a1, x) - np.dot(x, b1), scale * c1, decimal=4)