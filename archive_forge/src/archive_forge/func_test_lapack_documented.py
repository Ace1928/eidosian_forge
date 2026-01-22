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
def test_lapack_documented():
    """Test that all entries are in the doc."""
    if lapack.__doc__ is None:
        pytest.skip('lapack.__doc__ is None')
    names = set(lapack.__doc__.split())
    ignore_list = {'absolute_import', 'clapack', 'division', 'find_best_lapack_type', 'flapack', 'print_function', 'HAS_ILP64'}
    missing = list()
    for name in dir(lapack):
        if not name.startswith('_') and name not in ignore_list and (name not in names):
            missing.append(name)
    assert missing == [], 'Name(s) missing from lapack.__doc__ or ignore_list'