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
def test_minimize_scalar_custom(self):

    def custmin(fun, bracket, args=(), maxfev=None, stepsize=0.1, maxiter=100, callback=None, **options):
        bestx = (bracket[1] + bracket[0]) / 2.0
        besty = fun(bestx)
        funcalls = 1
        niter = 0
        improved = True
        stop = False
        while improved and (not stop) and (niter < maxiter):
            improved = False
            niter += 1
            for testx in [bestx - stepsize, bestx + stepsize]:
                testy = fun(testx, *args)
                funcalls += 1
                if testy < besty:
                    besty = testy
                    bestx = testx
                    improved = True
            if callback is not None:
                callback(bestx)
            if maxfev is not None and funcalls >= maxfev:
                stop = True
                break
        return optimize.OptimizeResult(fun=besty, x=bestx, nit=niter, nfev=funcalls, success=niter > 1)
    res = optimize.minimize_scalar(self.fun, bracket=(0, 4), method=custmin, options=dict(stepsize=0.05))
    assert_allclose(res.x, self.solution, atol=1e-06)