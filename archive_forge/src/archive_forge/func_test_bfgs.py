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
def test_bfgs(self):
    if self.use_wrapper:
        opts = {'maxiter': self.maxiter, 'disp': self.disp, 'return_all': False}
        res = optimize.minimize(self.func, self.startparams, jac=self.grad, method='BFGS', args=(), options=opts)
        params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = (res['x'], res['fun'], res['jac'], res['hess_inv'], res['nfev'], res['njev'], res['status'])
    else:
        retval = optimize.fmin_bfgs(self.func, self.startparams, self.grad, args=(), maxiter=self.maxiter, full_output=True, disp=self.disp, retall=False)
        params, fopt, gopt, Hopt, func_calls, grad_calls, warnflag = retval
    assert_allclose(self.func(params), self.func(self.solution), atol=1e-06)
    assert self.funccalls == 10, self.funccalls
    assert self.gradcalls == 8, self.gradcalls
    assert_allclose(self.trace[6:8], [[0, -0.525060743, 0.487748473], [0, -0.524885582, 0.487530347]], atol=1e-14, rtol=1e-07)