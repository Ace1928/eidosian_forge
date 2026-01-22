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
@pytest.mark.slow
def test_4_1_known_f_min(self):
    """Test known function minima stopping criteria"""
    options = {'f_min': test4_1.expected_fun, 'f_tol': 1e-06, 'minimize_every_iter': True}
    run_test(test4_1, n=None, test_atol=1e-05, options=options, sampling_method='simplicial')