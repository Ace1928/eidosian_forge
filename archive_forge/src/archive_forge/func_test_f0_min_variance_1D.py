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
@pytest.mark.skip('Not a test')
def test_f0_min_variance_1D(self):
    """Return a minimum on a perfectly symmetric 1D problem, based on
            gh10538"""

    def fun(x):
        return x * (x - 1.0) * (x - 0.5)
    bounds = [(0, 1)]
    res = shgo(fun, bounds=bounds)
    ref = minimize_scalar(fun, bounds=bounds[0])
    assert res.success
    assert_allclose(res.fun, ref.fun)
    assert_allclose(res.x, ref.x, rtol=1e-06)