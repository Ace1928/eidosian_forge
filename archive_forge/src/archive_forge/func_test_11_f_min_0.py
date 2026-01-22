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
def test_11_f_min_0(self):
    """Test to cover the case where f_lowest == 0"""
    options = {'f_min': 0.0, 'disp': True}
    res = shgo(test1_2.f, test1_2.bounds, n=10, iters=None, options=options, sampling_method='sobol')
    numpy.testing.assert_equal(0, res.x[0])
    numpy.testing.assert_equal(0, res.x[1])