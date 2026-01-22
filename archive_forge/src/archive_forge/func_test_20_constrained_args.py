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
def test_20_constrained_args(self):
    """Test that constraints can be passed to arguments"""

    def eggholder(x):
        return -(x[1] + 47.0) * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0)))) - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))

    def f(x):
        return 24.55 * x[0] + 26.75 * x[1] + 39 * x[2] + 40.5 * x[3]
    bounds = [(0, 1.0)] * 4

    def g1_modified(x, i):
        return i * 2.3 * x[0] + i * 5.6 * x[1] + 11.1 * x[2] + 1.3 * x[3] - 5

    def g2(x):
        return 12 * x[0] + 11.9 * x[1] + 41.8 * x[2] + 52.1 * x[3] - 21 - 1.645 * numpy.sqrt(0.28 * x[0] ** 2 + 0.19 * x[1] ** 2 + 20.5 * x[2] ** 2 + 0.62 * x[3] ** 2)

    def h1(x):
        return x[0] + x[1] + x[2] + x[3] - 1
    cons = ({'type': 'ineq', 'fun': g1_modified, 'args': (0,)}, {'type': 'ineq', 'fun': g2}, {'type': 'eq', 'fun': h1})
    shgo(f, bounds, n=300, iters=1, constraints=cons)
    shgo(f, bounds, n=300, iters=1, constraints=cons, sampling_method='sobol')