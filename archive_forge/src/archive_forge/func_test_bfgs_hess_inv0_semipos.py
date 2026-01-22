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
def test_bfgs_hess_inv0_semipos(self):
    with pytest.raises(ValueError, match="'hess_inv0' matrix isn't positive definite."):
        x0 = np.array([1.3, 0.7, 0.8, 1.9, 1.2])
        hess_inv0 = np.eye(5)
        hess_inv0[0, 0] = 0
        opts = {'disp': self.disp, 'hess_inv0': hess_inv0}
        optimize.minimize(optimize.rosen, x0=x0, method='BFGS', args=(), options=opts)