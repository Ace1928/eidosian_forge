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
def test_21_arg_tuple_sobol(self):
    """shgo used to raise an error when passing `args` with Sobol sampling
        # see gh-12114. check that this is resolved"""

    def fun(x, k):
        return x[0] ** k
    constraints = {'type': 'ineq', 'fun': lambda x: x[0] - 1}
    bounds = [(0, 10)]
    res = shgo(fun, bounds, args=(1,), constraints=constraints, sampling_method='sobol')
    ref = minimize(fun, numpy.zeros(1), bounds=bounds, args=(1,), constraints=constraints)
    assert res.success
    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x)