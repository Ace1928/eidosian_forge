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
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.parametrize('method', MINIMIZE_METHODS_NEW_CB)
@pytest.mark.parametrize('new_cb_interface', [0, 1, 2])
def test_callback_stopiteration(self, method, new_cb_interface):

    def f(x):
        f.flag = False
        return optimize.rosen(x)
    f.flag = False

    def g(x):
        f.flag = False
        return optimize.rosen_der(x)

    def h(x):
        f.flag = False
        return optimize.rosen_hess(x)
    maxiter = 5
    if new_cb_interface == 1:

        def callback_interface(*, intermediate_result):
            assert intermediate_result.fun == f(intermediate_result.x)
            callback()
    elif new_cb_interface == 2:

        class Callback:

            def __call__(self, intermediate_result: OptimizeResult):
                assert intermediate_result.fun == f(intermediate_result.x)
                callback()
        callback_interface = Callback()
    else:

        def callback_interface(xk, *args):
            callback()

    def callback():
        callback.i += 1
        callback.flag = False
        if callback.i == maxiter:
            callback.flag = True
            raise StopIteration()
    callback.i = 0
    callback.flag = False
    kwargs = {'x0': [1.1] * 5, 'method': method, 'fun': f, 'jac': g, 'hess': h}
    res = optimize.minimize(**kwargs, callback=callback_interface)
    if method == 'nelder-mead':
        maxiter = maxiter + 1
    ref = optimize.minimize(**kwargs, options={'maxiter': maxiter})
    assert res.fun == ref.fun
    assert_equal(res.x, ref.x)
    assert res.nit == ref.nit == maxiter
    assert res.status == (3 if method == 'trust-constr' else 99)