import itertools
import platform
import numpy as np
from numpy.testing import (assert_allclose, assert_equal,
import pytest
from pytest import raises as assert_raises
from scipy import optimize
from scipy.optimize._minimize import Bounds, NonlinearConstraint
from scipy.optimize._minimize import (MINIMIZE_METHODS,
from scipy.optimize._linprog import LINPROG_METHODS
from scipy.optimize._root import ROOT_METHODS
from scipy.optimize._root_scalar import ROOT_SCALAR_METHODS
from scipy.optimize._qap import QUADRATIC_ASSIGNMENT_METHODS
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize._optimize import MemoizeJac, show_options, OptimizeResult
from scipy.optimize import rosen, rosen_der, rosen_hess
from scipy.sparse import (coo_matrix, csc_matrix, csr_matrix, coo_array,
@pytest.mark.parametrize('method', ['Newton-CG', 'trust-constr'])
@pytest.mark.parametrize('sparse_type', [coo_matrix, csc_matrix, csr_matrix, coo_array, csr_array, csc_array])
def test_sparse_hessian(method, sparse_type):

    def sparse_rosen_hess(x):
        return sparse_type(rosen_hess(x))
    x0 = [2.0, 2.0]
    res_sparse = optimize.minimize(rosen, x0, method=method, jac=rosen_der, hess=sparse_rosen_hess)
    res_dense = optimize.minimize(rosen, x0, method=method, jac=rosen_der, hess=rosen_hess)
    assert_allclose(res_dense.fun, res_sparse.fun)
    assert_allclose(res_dense.x, res_sparse.x)
    assert res_dense.nfev == res_sparse.nfev
    assert res_dense.njev == res_sparse.njev
    assert res_dense.nhev == res_sparse.nhev