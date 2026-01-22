from numpy.testing import assert_, assert_equal
import pytest
from pytest import raises as assert_raises, warns as assert_warns
import numpy as np
from scipy.optimize import root
def test_gh_10370(self):

    def fun(x, ignored):
        return [3 * x[0] - 0.25 * x[1] ** 2 + 10, 0.1 * x[0] ** 2 + 5 * x[1] - 2]

    def grad(x, ignored):
        return [[3, 0.5 * x[1]], [0.2 * x[0], 5]]

    def fun_grad(x, ignored):
        return (fun(x, ignored), grad(x, ignored))
    x0 = np.zeros(2)
    ref = root(fun, x0, args=(1,), method='krylov')
    message = 'Method krylov does not use the jacobian'
    with assert_warns(RuntimeWarning, match=message):
        res1 = root(fun, x0, args=(1,), method='krylov', jac=grad)
    with assert_warns(RuntimeWarning, match=message):
        res2 = root(fun_grad, x0, args=(1,), method='krylov', jac=True)
    assert_equal(res1.x, ref.x)
    assert_equal(res2.x, ref.x)
    assert res1.success is res2.success is ref.success is True