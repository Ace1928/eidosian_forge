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
def test_sgesdd_lwork_bug_workaround():
    sgesdd_lwork = get_lapack_funcs('gesdd_lwork', dtype=np.float32, ilp64='preferred')
    n = 9537
    lwork = _compute_lwork(sgesdd_lwork, n, n, compute_uv=True, full_matrices=True)
    assert lwork == 272929888 or lwork == 272929920