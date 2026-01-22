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
def test_3_1_no_min_pool_sobol(self):
    """Check that the routine stops when no minimiser is found
           after maximum specified function evaluations"""
    options = {'maxfev': 10, 'disp': True}
    res = shgo(test_table.f, test_table.bounds, n=3, options=options, sampling_method='sobol')
    numpy.testing.assert_equal(False, res.success)
    numpy.testing.assert_equal(12, res.nfev)