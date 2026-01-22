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
def test_lartg():
    for dtype in 'fdFD':
        lartg = get_lapack_funcs('lartg', dtype=dtype)
        f = np.array(3, dtype)
        g = np.array(4, dtype)
        if np.iscomplexobj(g):
            g *= 1j
        cs, sn, r = lartg(f, g)
        assert_allclose(cs, 3.0 / 5.0)
        assert_allclose(r, 5.0)
        if np.iscomplexobj(g):
            assert_allclose(sn, -4j / 5.0)
            assert_(isinstance(r, complex))
            assert_(isinstance(cs, float))
        else:
            assert_allclose(sn, 4.0 / 5.0)