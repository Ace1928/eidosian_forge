import logging
import sys
import numpy
import numpy as np
import time
from multiprocessing import Pool
from numpy.testing import assert_allclose, IS_PYPY
import pytest
from pytest import raises as assert_raises, warns
from scipy.optimize import (shgo, Bounds, minimize_scalar, minimize, rosen,
from scipy.optimize._constraints import new_constraint_to_old
from scipy.optimize._shgo import SHGO
@pytest.mark.parametrize('derivative', ['jac', 'hess', 'hessp'])
def test_21_2_derivative_options(self, derivative):
    """shgo used to raise an error when passing `options` with 'jac'
        # see gh-12963. check that this is resolved
        """

    def objective(x):
        return 3 * x[0] * x[0] + 2 * x[0] + 5

    def gradient(x):
        return 6 * x[0] + 2

    def hess(x):
        return 6

    def hessp(x, p):
        return 6 * p
    derivative_funcs = {'jac': gradient, 'hess': hess, 'hessp': hessp}
    options = {derivative: derivative_funcs[derivative]}
    minimizer_kwargs = {'method': 'trust-constr'}
    bounds = [(-100, 100)]
    res = shgo(objective, bounds, minimizer_kwargs=minimizer_kwargs, options=options)
    ref = minimize(objective, x0=[0], bounds=bounds, **minimizer_kwargs, **options)
    assert res.success
    numpy.testing.assert_allclose(res.fun, ref.fun)
    numpy.testing.assert_allclose(res.x, ref.x)