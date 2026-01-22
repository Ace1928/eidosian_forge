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
def test_gehrd(self):
    a = [[-149, -50, -154], [537, 180, 546], [-27, -9, -25]]
    for p in 'd':
        f = getattr(flapack, p + 'gehrd', None)
        if f is None:
            continue
        ht, tau, info = f(a)
        assert_(not info, repr(info))