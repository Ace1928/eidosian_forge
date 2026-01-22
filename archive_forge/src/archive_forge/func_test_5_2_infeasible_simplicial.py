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
def test_5_2_infeasible_simplicial(self):
    """Ensures the algorithm terminates on infeasible problems
           after maxev is exceeded."""
    options = {'maxev': 1000, 'disp': False}
    res = shgo(test_infeasible.f, test_infeasible.bounds, constraints=test_infeasible.cons, n=100, options=options, sampling_method='simplicial')
    numpy.testing.assert_equal(False, res.success)