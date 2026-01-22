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
def test_2_2_sobol_iter(self):
    """Iterative Sobol sampling on TestFunction 2 (univariate)"""
    res = shgo(test2_1.f, test2_1.bounds, constraints=test2_1.cons, n=None, iters=1, sampling_method='sobol')
    numpy.testing.assert_allclose(res.x, test2_1.expected_x, rtol=1e-05, atol=1e-05)
    numpy.testing.assert_allclose(res.fun, test2_1.expected_fun, atol=1e-05)